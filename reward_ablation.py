"""离线奖励项归因：用已落盘的真实评测轨迹，量化每个奖励分项提供了多少训练信号。

## 为什么要单独做这个

电商场景的复合奖励有 4 个正分项：

    r = EM + 0.1·有引用 + 0.2·引用真实 + 0.2·引用命中gold − 0.05·min(搜索次数, 4)

评测结果显示 **`citation/source_gold` 在 base 与 step-20 上都是 92.9%，完全不随训练变化**。
一个自然的怀疑是：这 0.2 权重「设计了但没生效」。本脚本不重新训练，而是对
`eval_result/` 里已经保存的逐题轨迹做反事实重打分，从三个可观测角度给出证据：

1. **触发率**：有多少比例的题真正吃到了这个分项；
2. **与 EM 的共线性（φ 系数）**：若某分项是 EM 的确定性函数，那么在 GRPO 的
   组内中心化（advantage = reward − 组均值）之后，它的贡献被 EM 完全吸收，
   不提供任何独立梯度；
3. **对奖励景观的塑形能力**：把该分项权重置 0 后，跨题奖励排序变化多少（Kendall τ-b）。

## 与 GRPO 的关系（重要）

`rollout.assign_group_advantages` 用的是最简单的组内中心化，没有除标准差：

    advantage_i = reward_i − mean(reward_group)

因此**任何在同一组内恒为常数的分项，对 advantage 的贡献都是 0**。
评测集每题只有 1 条轨迹（group_size=1），无法直接测组内方差，所以这里用
「跨题触发率 + 与 EM 的共线性」作为可观测代理，并明确标注为代理指标。

用法：

    uv run python reward_ablation.py \
        --eval eval_result/ecom_eval_base.jsonl eval_result/ecom_eval_step20.jsonl
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from reward import DEFAULT_WEIGHTS, REWARD_PRESETS, RewardWeights, score_answer

COMPONENTS = ["exact_match", "has_source", "source_retrieved", "source_gold", "search_penalty"]
# 参与「被 EM 吸收」判定的正分项（search_penalty 是负项，不做该判定）
POSITIVE_COMPONENTS = ["has_source", "source_retrieved", "source_gold"]
# φ 系数达到该阈值即认为该分项是 EM 的确定性函数
ABSORPTION_PHI_THRESHOLD = 0.95


def load_trajectories(path: Path) -> list[dict]:
    """读取一个评测 JSONL，只保留逐题轨迹行。"""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("type") == "trajectory":
                rows.append(record)
    if not rows:
        raise ValueError(f"{path} 中没有 trajectory 记录")
    return rows


def load_gold_docs(dataset: Path | None) -> dict[str, list[str]]:
    """从数据集 JSONL 读 id → gold_docs 映射。

    评测轨迹里只保存了题号，没有保存 gold_docs（避免评测文件里出现"答案"信息）。
    归因需要按 gold 命中重算 reward，因此在这里按题号回填。
    """
    if dataset is None or not dataset.exists():
        return {}
    lookup: dict[str, list[str]] = {}
    with dataset.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            lookup[str(row["id"])] = list(row.get("gold_docs") or [])
    return lookup


def phi_coefficient(labels: list[bool], targets: list[bool]) -> float:
    """计算两个二值变量的 φ 相关系数；任意边缘为 0 时返回 0。"""
    n = len(labels)
    if n == 0:
        return 0.0
    n11 = sum(a and b for a, b in zip(labels, targets, strict=True))
    n10 = sum(a and not b for a, b in zip(labels, targets, strict=True))
    n01 = sum((not a) and b for a, b in zip(labels, targets, strict=True))
    n00 = n - n11 - n10 - n01
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return 0.0 if denominator == 0.0 else (n11 * n00 - n10 * n01) / denominator


def kendall_tau_b(left: list[float], right: list[float]) -> float:
    """Kendall τ-b：允许并列值的秩相关系数。"""
    n = len(left)
    if n < 2:
        return 1.0
    concordant = discordant = ties_left = ties_right = 0
    for i in range(n):
        for j in range(i + 1, n):
            dl = left[i] - left[j]
            dr = right[i] - right[j]
            if dl == 0.0 and dr == 0.0:
                ties_left += 1
                ties_right += 1
            elif dl == 0.0:
                ties_left += 1
            elif dr == 0.0:
                ties_right += 1
            elif (dl > 0) == (dr > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = concordant + discordant
    denominator = math.sqrt((pairs + ties_left) * (pairs + ties_right))
    return (concordant - discordant) / denominator if denominator else 1.0


def compare_scores(baseline: list[float], variant: list[float]) -> dict[str, int]:
    """比较两套打分：多少题数值变了、多少题对的相对顺序变了。

    GRPO 的组内中心化 advantage = reward − 组均值，因此**只做全局常数偏移的
    奖励项对 advantage 零贡献**。判据是：数值变了但题对的相对顺序一对都没变
    （`order_changed == 0`）——这意味着该项在所有题上加的是同一个常数（或干脆没加）。
    """
    n = len(baseline)
    value_changed = sum(
        1 for a, b in zip(baseline, variant, strict=True) if abs(a - b) > 1e-9
    )
    flipped = order_changed = 0
    for i in range(n):
        for j in range(i + 1, n):
            before = baseline[i] - baseline[j]
            after = variant[i] - variant[j]
            before_sign = 0 if abs(before) < 1e-9 else (1 if before > 0 else -1)
            after_sign = 0 if abs(after) < 1e-9 else (1 if after > 0 else -1)
            if before_sign != after_sign:
                order_changed += 1
                if before_sign != 0 and after_sign != 0:
                    flipped += 1
    return {
        "value_changed": value_changed,
        "order_changed_pairs": order_changed,
        "flipped_pairs": flipped,
    }


def confusion_with_em(rows: list[dict]) -> dict[str, int]:
    """EM × source_gold 列联表，刻画「引用项到底在惩罚谁」。"""
    table = {"em1_gold1": 0, "em1_gold0": 0, "em0_gold1": 0, "em0_gold0": 0}
    for row in rows:
        key = f"em{int(bool(row.get('exact_match')))}_gold{int(bool(row.get('source_gold')))}"
        table[key] += 1
    return table


def rescore(
    rows: list[dict], weights: RewardWeights, gold_lookup: dict[str, list[str]]
) -> list[float]:
    """用指定权重对每题的 final_text 重新打分。"""
    scores: list[float] = []
    for row in rows:
        gold = gold_lookup.get(str(row["id"]))
        result = score_answer(
            row["final_text"],
            row["answers"],
            retrieved_doc_ids=row.get("retrieved_doc_ids") or [],
            gold_doc_ids=gold or None,
            search_calls=int(row.get("search_calls") or 0),
            weights=weights,
        )
        scores.append(result.reward)
    return scores


def component_series(
    row: dict, gold_lookup: dict[str, list[str]], weights: RewardWeights
) -> dict[str, float]:
    """把一道题的复合奖励拆成逐分项贡献（口径与 score_answer 完全一致）。"""
    gold = gold_lookup.get(str(row["id"]))
    result = score_answer(
        row["final_text"],
        row["answers"],
        retrieved_doc_ids=row.get("retrieved_doc_ids") or [],
        gold_doc_ids=gold or None,
        search_calls=int(row.get("search_calls") or 0),
        weights=weights,
    )
    if result.answer is None:  # 格式非法：只有格式惩罚
        return {
            "exact_match": 0.0,
            "has_source": 0.0,
            "source_retrieved": 0.0,
            "source_gold": 0.0,
            "search_penalty": -0.1,
        }
    calls = int(row.get("search_calls") or 0)
    return {
        "exact_match": weights.exact_match * float(result.exact_match),
        "has_source": weights.has_source * float(result.has_source),
        "source_retrieved": weights.source_retrieved * float(result.source_retrieved),
        "source_gold": weights.source_gold * float(result.source_gold),
        "search_penalty": -weights.search_penalty
        * min(calls, weights.search_penalty_cap),
    }


def trigger_flags(row: dict, component: str) -> bool:
    """取某个分项在该题上是否被触发。"""
    if component == "search_penalty":
        return int(row.get("search_calls") or 0) > 0
    return bool(row.get(component))


def conditional_rate(flags: list[bool], exact: list[bool], target: bool) -> float:
    """P(分项触发 | EM == target)。"""
    subset = [flag for flag, em in zip(flags, exact, strict=True) if em is target]
    return sum(subset) / len(subset) if subset else float("nan")


def rate_of(values) -> float:
    """把布尔序列转成比例。"""
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def analyze(name: str, rows: list[dict], gold_lookup: dict[str, list[str]]) -> dict:
    """对一个评测文件做全部分项归因分析。"""
    exact = [bool(r.get("exact_match")) for r in rows]
    full_scores = rescore(rows, REWARD_PRESETS["full"], gold_lookup)
    saved = [float(r.get("reward") or 0.0) for r in rows]
    max_drift = max(
        (abs(a - b) for a, b in zip(saved, full_scores, strict=True)), default=0.0
    )

    series: dict[str, list[float]] = {c: [] for c in COMPONENTS}
    for row in rows:
        per_row = component_series(row, gold_lookup, DEFAULT_WEIGHTS)
        for component in COMPONENTS:
            series[component].append(per_row[component])

    term_rows: list[dict] = []
    for component in COMPONENTS:
        flags = [trigger_flags(r, component) for r in rows]
        values = series[component]
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        phi = phi_coefficient(flags, exact)
        term_rows.append(
            {
                "component": component,
                "trigger_rate": sum(flags) / len(flags),
                "mean_contribution": mean_value,
                "std_contribution": math.sqrt(variance),
                "P_trigger_given_em1": conditional_rate(flags, exact, True),
                "P_trigger_given_em0": conditional_rate(flags, exact, False),
                "phi_with_em": phi,
                "absorbed_by_em": component in POSITIVE_COMPONENTS
                and phi >= ABSORPTION_PHI_THRESHOLD,
            }
        )

    counterfactual: list[dict] = []
    full_mean = sum(full_scores) / len(full_scores)
    for preset_name, weights in sorted(REWARD_PRESETS.items()):
        if preset_name == "full":
            continue
        scores = rescore(rows, weights, gold_lookup)
        mean_score = sum(scores) / len(scores)
        comparison = compare_scores(full_scores, scores)
        counterfactual.append(
            {
                "preset": preset_name,
                "mean_reward": mean_score,
                "delta_vs_full": mean_score - full_mean,
                "kendall_tau_b_vs_full": kendall_tau_b(full_scores, scores),
                **comparison,
                # 数值变了但题对顺序一对没变 → 该项只贡献了全局常数偏移
                "constant_shift_only": comparison["value_changed"] > 0
                and comparison["order_changed_pairs"] == 0,
            }
        )

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("data_source") or "unknown")].append(row)
    source_rows = [
        {
            "data_source": source,
            "n": len(items),
            "em": rate_of(bool(i.get("exact_match")) for i in items),
            "source_gold": rate_of(bool(i.get("source_gold")) for i in items),
            "has_source": rate_of(bool(i.get("has_source")) for i in items),
            "mean_search_calls": sum(int(i.get("search_calls") or 0) for i in items)
            / len(items),
        }
        for source, items in sorted(by_source.items())
    ]

    return {
        "name": name,
        "n": len(rows),
        "max_reward_drift_vs_saved": max_drift,
        "terms": term_rows,
        "counterfactual": counterfactual,
        "by_source": source_rows,
        "confusion_em_x_source_gold": confusion_with_em(rows),
    }


def print_report(reports: list[dict]) -> None:
    """把归因结果打印成可读表格。"""
    for report in reports:
        print(f"\n{'=' * 78}\n## {report['name']}   n={report['n']}")
        print(
            f"完整性校验：落盘 reward 与按 full 权重重算的最大偏差 = "
            f"{report['max_reward_drift_vs_saved']:.6f}"
        )
        if report["max_reward_drift_vs_saved"] > 1e-6:
            print("  ⚠️ 偏差非零：落盘 reward 的权重与当前 full 配置不一致")

        print("\n### 分项归因")
        print(
            f"{'分项':<18}{'触发率':>9}{'平均贡献':>10}{'std':>8}"
            f"{'P(·|EM=1)':>12}{'P(·|EM=0)':>12}{'φ(·,EM)':>9}  判定"
        )
        for row in report["terms"]:
            if row["absorbed_by_em"]:
                verdict = "被 EM 吸收，无独立梯度"
            elif row["std_contribution"] == 0.0:
                verdict = "无区分度"
            else:
                verdict = "有独立变化"
            print(
                f"{row['component']:<18}{row['trigger_rate']:>8.2%}"
                f"{row['mean_contribution']:>10.3f}{row['std_contribution']:>8.3f}"
                f"{row['P_trigger_given_em1']:>12.2%}{row['P_trigger_given_em0']:>12.2%}"
                f"{row['phi_with_em']:>9.3f}  {verdict}"
            )

        print("\n### 反事实重打分（相对 full 基准）")
        print(
            f"{'预设':<22}{'平均奖励':>10}{'Δ奖励':>10}{'数值变动题数':>13}"
            f"{'顺序变动题对':>13}  判定"
        )
        for row in report["counterfactual"]:
            if row["constant_shift_only"]:
                verdict = "仅全局常数偏移 → 组内零梯度"
            elif row["order_changed_pairs"] == 0:
                verdict = "无影响"
            else:
                verdict = "改变奖励景观"
            print(
                f"{row['preset']:<22}{row['mean_reward']:>10.3f}"
                f"{row['delta_vs_full']:>+10.3f}{row['value_changed']:>13}"
                f"{row['order_changed_pairs']:>13}  {verdict}"
            )

        confusion = report["confusion_em_x_source_gold"]
        print("\n### EM × 引用命中 gold 列联表（引用项到底在惩罚谁）")
        print(f"  EM=1 且 引用对 : {confusion['em1_gold1']:>3}")
        print(f"  EM=1 但 引用错 : {confusion['em1_gold0']:>3}   ← 引用项唯一真正惩罚的「答对却引错」")
        print(f"  EM=0 但 引用对 : {confusion['em0_gold1']:>3}   ← 引用项无法帮助的「引对却答错」")
        print(f"  EM=0 且 引用错 : {confusion['em0_gold0']:>3}")

        print("\n### 按题型分解")
        print(
            f"{'data_source':<26}{'n':>4}{'EM':>9}{'source_gold':>13}"
            f"{'has_source':>12}{'搜索次数':>10}"
        )
        for row in report["by_source"]:
            print(
                f"{row['data_source']:<26}{row['n']:>4}{row['em']:>9.2%}"
                f"{row['source_gold']:>13.2%}{row['has_source']:>12.2%}"
                f"{row['mean_search_calls']:>10.2f}"
            )


def print_cross_checkpoint(reports: list[dict]) -> None:
    """跨 checkpoint 对照：哪些分项的触发率完全不动。"""
    if len(reports) < 2:
        return
    first, last = reports[0], reports[-1]
    print(f"\n{'=' * 78}\n## 跨 checkpoint 对照：{first['name']} → {last['name']}")
    print(f"{'分项':<18}{'触发率(前)':>12}{'触发率(后)':>12}{'Δ':>10}  判定")
    before_table = {term["component"]: term for term in first["terms"]}
    for term in last["terms"]:
        component = term["component"]
        before = before_table[component]["trigger_rate"]
        after = term["trigger_rate"]
        delta = after - before
        verdict = "完全不动（已饱和 / 未生效）" if abs(delta) < 1e-9 else "有变化"
        print(f"{component:<18}{before:>12.2%}{after:>12.2%}{delta:>+10.2%}  {verdict}")


def parse_args() -> argparse.Namespace:
    """解析评测文件路径与输出路径。"""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        nargs="+",
        default=[
            base / "eval_result" / "ecom_eval_base.jsonl",
            base / "eval_result" / "ecom_eval_step20.jsonl",
        ],
        help="待分析的评测 JSONL（可传多个，默认分析电商 base / step-20）",
    )
    parser.add_argument(
        "--gold-docs",
        type=Path,
        default=base / "datasets" / "ecommerce" / "dev.jsonl",
        help="提供 id → gold_docs 映射的数据集 JSONL（评测轨迹本身不含 gold_docs）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "eval_result" / "reward_ablation.json",
        help="归因结果 JSON 输出路径",
    )
    return parser.parse_args()


def main() -> None:
    """跑全部评测文件的归因并落盘。"""
    args = parse_args()
    gold_lookup = load_gold_docs(args.gold_docs)
    if not gold_lookup:
        raise ValueError(f"未能从 {args.gold_docs} 读到 gold_docs，无法做引用项归因")
    reports = [
        analyze(path.stem, load_trajectories(path), gold_lookup) for path in args.eval
    ]
    print_report(reports)
    print_cross_checkpoint(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(reports, file, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
