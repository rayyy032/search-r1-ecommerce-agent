"""SFT 冷启动：从 base 模型做拒绝采样，构造「答案对 + 引用对」的高质量轨迹再监督微调。

## 为什么要有这一步

原实验的链路是 `base → RL`，缺了中间那一段，于是被问「你为什么要做 RL？SFT 做到头了吗？」
时答不上来。本项目补上标准的三段式：

    base  →  SFT（拒绝采样冷启动）  →  RL（GRPO + 引用可溯源奖励）

设计上参考 Alibaba-NLP/DeepResearch 的 WebDancer / WebSailor 两阶段 recipe：
先用当前策略对同一道题采样多条轨迹，只保留**完全正确**的（拒绝采样 / rejection sampling），
把保留下来的轨迹当作示范做 SFT；然后再用 RL 去攻那些拒绝采样筛不出来的难题。

## 拒绝采样通过率本身就是难度分层

本脚本除了产出 SFT 权重，还会写出 `rejection_stats.json`：**每道题的轨迹通过率**。
这个通过率是可免费得到的课程学习（curriculum）信号：

- 通过率 = 0 的题：base 完全做不出 → SFT 学不到，留给 RL 主攻
- 通过率 ∈ (0, 1)：有对有错 → SFT 的黄金样本（既有正例，题目又不至于太容易）
- 通过率 = 1 的题：base 已经会 → 对 SFT 和 RL 都几乎无增益

这正是「数据环节里你自己设计的那一步」，而不是「直接下载开源数据集」。

## 用法

    SEARCH_R1_SCENE=ecommerce uv run python sft.py \
        --data datasets/ecommerce/train.jsonl \
        --search-backend ecommerce \
        --questions 60 --samples-per-question 4 \
        --min-reward 1.0 --epochs 2 \
        --run-name ecom-sft-coldstart

产出 `sft_runs/<run-name>/` 下的 state 与 sampler weights 路径，然后接 RL：

    SEARCH_R1_SCENE=ecommerce uv run python train.py \
        --resume-state '<上一步输出的 state 路径>' \
        --max-steps 20 --search-backend ecommerce \
        --run-name ecom-sft-then-rl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

from data import shuffled_examples, take_batch
from rollout import RolloutConfig, Trajectory, rollout_batch
from search import SEARCH_BACKENDS, create_search_client, resolve_search_concurrency, resolve_search_timeout

MAX_TRAIN_CONTEXT_TOKENS = 8192  # 与 train.py 对齐：单个 Datum 允许的最大 token 数
MICRO_BATCH_MAX_ITEMS = 8  # 一个 micro-batch 最多容纳的 Datum 数
MICRO_BATCH_MAX_PADDED_TOKENS = 32_000  # 单批 padding 矩形允许的最大 token 数


def build_sft_datum(trajectory: Trajectory) -> trio.Datum | None:
    """把一条完整轨迹转成只对 assistant token 计权的 cross_entropy Datum。

    与 train.py 的 build_datum 共用同一套「前缀扩展」token 拼装逻辑（RL 用 advantage
    加权，SFT 用 weights 掩码），保证两段训练看到完全一样的序列。返回 None 表示
    该轨迹超长或没有 assistant token，应被跳过。
    """
    full_tokens: list[int] = []
    token_weights: list[float] = []
    for turn_index, turn in enumerate(trajectory.turns):
        if len(turn.completion_tokens) == 0:
            continue
        if turn_index == 0:
            delta_observation = turn.prompt_tokens
        elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta_observation = turn.prompt_tokens[len(full_tokens) :]
        else:
            # 前缀不一致说明轨迹被截断过，无法安全对齐，直接丢弃
            return None
        # observation（用户提问 / 工具返回）不参与 loss，权重 0
        full_tokens.extend(delta_observation)
        token_weights.extend([0.0] * len(delta_observation))
        # assistant 生成内容参与 loss，权重 1
        full_tokens.extend(turn.completion_tokens)
        token_weights.extend([1.0] * len(turn.completion_tokens))

    if not full_tokens or sum(token_weights) == 0.0:
        return None
    # 这里显式右移，与 train.py 一致（PyTRIO 的 auto_shift 默认关闭）
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    weights = token_weights[1:]
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
        return None
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "weights": np.asarray(weights, dtype=np.float32),
        },
    )


def pack_micro_batches(datums: list[trio.Datum]) -> list[list[trio.Datum]]:
    """按 padding 后的矩形面积做 first-fit decreasing 装箱，控制显存峰值。"""
    sized = sorted(
        ((len(datum.model_input), datum) for datum in datums),
        key=lambda pair: pair[0],
        reverse=True,
    )
    batches: list[list[trio.Datum]] = []
    batch_max_tokens: list[int] = []
    for size, datum in sized:
        for index, batch in enumerate(batches):
            next_items = len(batch) + 1
            next_max = max(batch_max_tokens[index], size)
            if (
                next_items <= MICRO_BATCH_MAX_ITEMS
                and next_items * next_max <= MICRO_BATCH_MAX_PADDED_TOKENS
            ):
                batch.append(datum)
                batch_max_tokens[index] = next_max
                break
        else:
            batches.append([datum])
            batch_max_tokens.append(size)
    return batches


def collect_rejections(
    sampling_client,
    tokenizer,
    search_client,
    examples,
    config: RolloutConfig,
    min_reward: float,
) -> tuple[list[Trajectory], list[dict]]:
    """对一批题做拒绝采样，返回保留的轨迹与逐题统计。"""
    trajectories = rollout_batch(sampling_client, tokenizer, search_client, examples, config)
    kept = [t for t in trajectories if t.reward >= min_reward]

    stats: list[dict] = []
    for example in examples:
        group = [t for t in trajectories if t.example.id == example.id]
        if not group:
            continue
        accepted = sum(1 for t in group if t.reward >= min_reward)
        stats.append(
            {
                "id": example.id,
                "data_source": example.data_source,
                "samples": len(group),
                "accepted": accepted,
                "acceptance_rate": accepted / len(group),
                "mean_reward": sum(t.reward for t in group) / len(group),
            }
        )
    return kept, stats


def parse_args() -> argparse.Namespace:
    """解析 SFT 数据、拒绝采样与训练超参。"""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=base_dir / "datasets" / "ecommerce" / "train.jsonl")
    parser.add_argument("--questions", type=int, default=60, help="使用多少道训练题做拒绝采样；0 表示全部")
    parser.add_argument("--questions-per-batch", type=int, default=8, help="每轮拒绝采样推进的题数")
    parser.add_argument("--samples-per-question", type=int, default=4, help="每道题采样多少条候选轨迹")
    parser.add_argument("--min-reward", type=float, default=1.0, help="拒绝采样阈值；≥1.0 等价于『答案对且引用全对』")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--search-backend", choices=SEARCH_BACKENDS, default="ecommerce")
    parser.add_argument("--search-concurrency", type=int, default=None)
    parser.add_argument("--search-model", default="deepseek-v4-flash")
    parser.add_argument("--search-timeout", type=float, default=None)
    parser.add_argument("--max-search-calls", type=int, default=4)
    parser.add_argument("--max-assistant-turns", type=int, default=6)
    parser.add_argument("--max-trajectory-tokens", type=int, default=8192)
    parser.add_argument("--max-assistant-tokens", type=int, default=1024)
    parser.add_argument("--max-tool-response-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8, help="拒绝采样温度；需要一定随机性才能筛出正例")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="ecom-sft-coldstart")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "sft_runs")
    parser.add_argument("--swanlab-project", default="agentic-rl-lab-search-r1")
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline", "disabled"], default="online")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """拒绝采样 → SFT → 保存可被 train.py --resume-state 接续的训练状态。"""
    args.search_concurrency = resolve_search_concurrency(args.search_backend, args.search_concurrency)
    args.search_timeout = resolve_search_timeout(args.search_backend, args.search_timeout)

    examples = shuffled_examples(args.data, args.seed)
    if args.questions > 0:
        examples = examples[: args.questions]
    if not examples:
        raise ValueError(f"训练数据为空：{args.data}")

    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model, rank=args.lora_rank, seed=args.seed
    )
    tokenizer = training_client.get_tokenizer()
    search_client = create_search_client(
        args.search_backend,
        Path(__file__).resolve().parent / ".env",
        model=args.search_model,
        timeout=args.search_timeout,
    )
    rollout_config = RolloutConfig(
        group_size=args.samples_per_question,
        max_search_calls=args.max_search_calls,
        max_assistant_turns=args.max_assistant_turns,
        max_trajectory_tokens=args.max_trajectory_tokens,
        max_assistant_tokens=args.max_assistant_tokens,
        max_tool_response_tokens=args.max_tool_response_tokens,
        search_concurrency=args.search_concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    run = swanlab.init(
        project=args.swanlab_project,
        name=args.run_name,
        mode=args.swanlab_mode,
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )

    try:
        # ---------------- 阶段 1：拒绝采样 ----------------
        all_stats: list[dict] = []
        kept_trajectories: list[Trajectory] = []
        batches = [
            examples[index : index + args.questions_per_batch]
            for index in range(0, len(examples), args.questions_per_batch)
        ]
        with tqdm(total=len(examples), desc="Rejection sampling", unit="question") as progress:
            for batch_index, batch in enumerate(batches, 1):
                sampling_client = training_client.save_weights_and_get_sampling_client()
                kept, stats = collect_rejections(
                    sampling_client, tokenizer, search_client, batch, rollout_config, args.min_reward
                )
                kept_trajectories.extend(kept)
                all_stats.extend(stats)
                progress.update(len(batch))
                progress.set_postfix(batch=f"{batch_index}/{len(batches)}", kept=len(kept_trajectories))

        datums: list[trio.Datum] = []
        skipped = 0
        for trajectory in kept_trajectories:
            datum = build_sft_datum(trajectory)
            if datum is None:
                skipped += 1
            else:
                datums.append(datum)
        if not datums:
            raise RuntimeError(
                "拒绝采样没有产出任何可用示范样本；请降低 --min-reward 或提高 --samples-per-question"
            )

        buckets = Counter(
            "hard(通过率=0)" if row["acceptance_rate"] == 0.0
            else "easy(通过率=1)" if row["acceptance_rate"] == 1.0
            else "golden(0<通过率<1)"
            for row in all_stats
        )
        total_sampled = sum(row["samples"] for row in all_stats)
        summary = {
            "run_name": args.run_name,
            "questions": len(all_stats),
            "samples_per_question": args.samples_per_question,
            "min_reward": args.min_reward,
            "trajectories_sampled": total_sampled,
            "trajectories_accepted": len(kept_trajectories),
            "acceptance_rate": len(kept_trajectories) / max(total_sampled, 1),
            "sft_datums": len(datums),
            "skipped_overlong": skipped,
            "difficulty_buckets": dict(buckets),
            "per_question": all_stats,
        }
        output_dir = args.output_dir / args.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = output_dir / "rejection_stats.json"
        stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"\n拒绝采样：采样 {total_sampled} 条 → 通过 {len(kept_trajectories)} 条 "
            f"({summary['acceptance_rate']:.1%})；"
            f"难度分层 {dict(buckets)}"
        )
        swanlab.log(
            {
                "sft/acceptance_rate": summary["acceptance_rate"],
                "sft/datums": float(len(datums)),
                "sft/hard_questions": float(buckets["hard(通过率=0)"]),
                "sft/golden_questions": float(buckets["golden(0<通过率<1)"]),
                "sft/easy_questions": float(buckets["easy(通过率=1)"]),
            },
            step=0,
        )

        # ---------------- 阶段 2：SFT ----------------
        micro_batches = pack_micro_batches(datums)
        print(
            f"SFT：{len(datums)} 条示范样本，{len(micro_batches)} 个 micro-batch，"
            f"{args.epochs} 个 epoch"
        )
        adam_params = trio.AdamParams(learning_rate=args.learning_rate)
        global_step = 0
        for epoch in range(args.epochs):
            epoch_losses: list[float] = []
            with tqdm(total=len(micro_batches), desc=f"SFT epoch {epoch + 1}/{args.epochs}", unit="batch") as progress:
                for batch in micro_batches:
                    result = training_client.forward_backward(batch, loss_fn="cross_entropy").result()
                    training_client.optim_step(adam_params).result()
                    metrics = dict(result.metrics)
                    loss = metrics.get("loss_mean", metrics.get("loss/mean"))
                    if isinstance(loss, (int, float)):
                        epoch_losses.append(float(loss))
                        swanlab.log({"sft/loss": float(loss), "sft/epoch": epoch + 1}, step=global_step)
                    progress.update(1)
                    global_step += 1
            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
            print(f"  epoch {epoch + 1}: mean loss = {mean_loss:.4f}")

        # ---------------- 阶段 3：保存，供 RL 接续 ----------------
        state = training_client.save_state(name=f"{args.run_name}-state").result()
        weights = training_client.save_weights_for_sampler(name=f"{args.run_name}-weights").result()
        print(f"\nSFT state（传给 train.py --resume-state）：{state.path}")
        print(f"SFT sampler weights（可直接 eval.py --model-path）：{weights.path}")
        print(f"拒绝采样统计：{stats_path}")
        run.finish()
    except Exception:
        run.finish()
        raise


if __name__ == "__main__":
    main(parse_args())
