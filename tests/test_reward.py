"""奖励函数与归因工具的自检用例。

运行：`uv run --extra dev pytest tests/ -q`

最后两个用例直接读 `eval_result/` 里已经落盘的 70 条真实评测轨迹，用
`datasets/ecommerce/dev.jsonl` 回填 gold_docs 后重算奖励，断言与落盘 reward
逐条完全相同——这相当于把「奖励公式」钉在一份真实实验产物上，
以后任何改动只要动了奖励语义，这两个用例就会红。"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reward import (  # noqa: E402
    DEFAULT_WEIGHTS,
    RewardWeights,
    resolve_weights,
    reward_contributions,
    score_answer,
)
from reward_ablation import (  # noqa: E402
    compare_scores,
    load_gold_docs,
    load_trajectories,
    rescore,
)

EVAL_DIR = REPO_ROOT / "eval_result"
DEV_DATA = REPO_ROOT / "datasets" / "ecommerce" / "dev.jsonl"
ECOMMERCE_EVALS = [EVAL_DIR / "ecom_eval_base.jsonl", EVAL_DIR / "ecom_eval_step20.jsonl"]


def test_default_weights_match_documented_formula():
    """默认权重下的奖励必须等于 README 里写的那个公式。"""
    result = score_answer(
        "Answer: 不能\nSource: P001, S001",
        ["不能"],
        retrieved_doc_ids=["P001", "S001"],
        gold_doc_ids=["S001"],
        search_calls=2,
    )
    expected = 1.0 + 0.1 + 0.2 + 0.2 - 0.05 * 2
    assert result.reward == pytest.approx(expected)
    assert (result.exact_match, result.has_source, result.source_retrieved, result.source_gold) == (
        True,
        True,
        True,
        True,
    )


def test_search_penalty_is_capped():
    """搜索惩罚封顶在 search_penalty_cap 次。"""
    weights = RewardWeights()
    for calls in (4, 5, 40):
        result = score_answer(
            "Answer: 不能\nSource: S001",
            ["不能"],
            retrieved_doc_ids=["S001"],
            gold_doc_ids=["S001"],
            search_calls=calls,
            weights=weights,
        )
        assert result.reward == pytest.approx(
            1.0 + 0.1 + 0.2 + 0.2 - weights.search_penalty * weights.search_penalty_cap
        )


def test_invalid_format_scores_negative():
    """缺少唯一 Answer 行时返回 -0.1 并标记格式非法。"""
    result = score_answer("答案是不能", ["不能"], gold_doc_ids=["S001"])
    assert result.reward == pytest.approx(-0.1)
    assert result.valid_format is False
    assert result.answer is None


def test_source_retrieved_requires_real_retrieval():
    """引用的编号没出现在本轮检索结果里时，source_retrieved 必须为假。"""
    result = score_answer(
        "Answer: 不能\nSource: P099",
        ["不能"],
        retrieved_doc_ids=["P001", "S001"],
        gold_doc_ids=["P099"],
    )
    assert result.source_retrieved is False
    assert result.source_gold is True  # gold 命中只看 gold_docs，与是否检索到无关


def test_wikipedia_scenario_without_gold_docs_is_em_only():
    """没有 gold_docs（英文 Wikipedia 场景）时退化为原版 EM 奖励。"""
    correct = score_answer("Answer: Paris", ["Paris"])
    wrong = score_answer("Answer: London", ["Paris"])
    assert correct.reward == pytest.approx(1.0)
    assert wrong.reward == pytest.approx(0.0)


def test_em_only_preset_removes_all_citation_weights():
    """em_only 预设下引用全对也不加分。"""
    text = "Answer: 不能\nSource: P001, S001"
    kwargs = {
        "retrieved_doc_ids": ["P001", "S001"],
        "gold_doc_ids": ["S001"],
        "search_calls": 2,
    }
    assert score_answer(text, ["不能"], **kwargs).reward == pytest.approx(1.4)
    assert score_answer(
        text, ["不能"], weights=resolve_weights("em_only"), **kwargs
    ).reward == pytest.approx(1.0)


def test_presets_are_single_variable():
    """每个消融预设相对 full 只能改动它名字里声明的那个字段。"""
    expected_field = {
        "no_source_gold": "source_gold",
        "no_source_retrieved": "source_retrieved",
        "no_has_source": "has_source",
        "no_search_penalty": "search_penalty",
    }
    full = DEFAULT_WEIGHTS.to_dict()
    for preset, field in expected_field.items():
        diff = {
            key
            for key, value in resolve_weights(preset).to_dict().items()
            if value != full[key]
        }
        assert diff == {field}, f"{preset} 意外改动了 {diff}"
    # 多个字段的预设单独检查
    for preset, fields in {
        "no_citation": {"has_source", "source_retrieved", "source_gold"},
        "em_plus_format": {"source_retrieved", "source_gold", "search_penalty"},
        "em_only": {
            "has_source",
            "source_retrieved",
            "source_gold",
            "search_penalty",
        },
    }.items():
        diff = {
            key
            for key, value in resolve_weights(preset).to_dict().items()
            if value != full[key]
        }
        assert diff == fields, f"{preset} 意外改动了 {diff}"


def test_unknown_preset_raises_with_options():
    """非法预设名要报出可选值，方便命令行自查。"""
    with pytest.raises(ValueError, match="no_source_gold"):
        resolve_weights("nope")


def test_contributions_sum_to_reward():
    """分项拆解之和必须精确等于总奖励。"""
    cases = [
        ("Answer: 不能\nSource: P001, S001", ["不能"], ["P001", "S001"], ["S001"], 2),
        ("Answer: 不能\nSource: P099", ["不能"], ["P001"], ["S001"], 0),
        ("Answer: 不支持\nSource: P001", ["不支持"], ["P001"], ["P001"], 4),
    ]
    for text, refs, retrieved, gold, calls in cases:
        result = score_answer(
            text, refs, retrieved_doc_ids=retrieved, gold_doc_ids=gold, search_calls=calls
        )
        parts = reward_contributions(result, calls)
        assert sum(parts.values()) == pytest.approx(result.reward)


def test_compare_scores_flags_pure_constant_shift():
    """全局常数偏移应被识别为『组内零梯度』：数值变了但题对顺序一对没变。"""
    baseline = [1.0, 2.0, 3.0, 4.0]
    shifted = [value + 0.3 for value in baseline]
    result = compare_scores(baseline, shifted)
    assert result["value_changed"] == 4
    assert result["order_changed_pairs"] == 0
    assert result["flipped_pairs"] == 0

    reshuffled = [4.0, 3.0, 2.0, 1.0]
    result = compare_scores(baseline, reshuffled)
    assert result["order_changed_pairs"] > 0
    assert result["flipped_pairs"] > 0


@pytest.mark.skipif(not DEV_DATA.exists(), reason="缺少电商数据集")
@pytest.mark.parametrize("eval_path", ECOMMERCE_EVALS)
def test_rescore_reproduces_committed_rewards(eval_path):
    """核心回归：用默认权重重算落盘轨迹，必须逐条复现已提交的 reward。"""
    if not eval_path.exists():
        pytest.skip(f"缺少 {eval_path.name}")
    gold_lookup = load_gold_docs(DEV_DATA)
    rows = load_trajectories(eval_path)
    recomputed = rescore(rows, DEFAULT_WEIGHTS, gold_lookup)
    saved = [float(row["reward"]) for row in rows]
    worst = max(abs(a - b) for a, b in zip(saved, recomputed, strict=True))
    assert worst < 1e-9, f"{eval_path.name} 奖励不一致，最大偏差 {worst}"


@pytest.mark.skipif(not DEV_DATA.exists(), reason="缺少电商数据集")
def test_every_eval_trajectory_has_gold_docs():
    """评测轨迹必须能按 id 在数据集里找到 gold_docs，否则引用项无法归因。"""
    gold_lookup = load_gold_docs(DEV_DATA)
    for eval_path in ECOMMERCE_EVALS:
        if not eval_path.exists():
            continue
        for row in load_trajectories(eval_path):
            assert gold_lookup.get(str(row["id"])), f"{row['id']} 在 dev 集里没有 gold_docs"
