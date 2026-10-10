# SFT 冷启动

原实验链路是 base → RL 两个点。被问"为什么要做 RL、SFT 做到头了吗"的时候，没有中间数据就答不上来。`sft.py` 补上标准三段式：

```
base  →  SFT（拒绝采样冷启动）  →  RL（GRPO + 可溯源奖励）
```

做法参考 Alibaba-NLP/DeepResearch 的 WebDancer / WebSailor 两阶段 recipe：用当前策略对同一道题采样多条轨迹，只保留完全正确的，把保留的轨迹当示范做 SFT，再用 RL 去攻那些筛不出来的难题。

```bash
SEARCH_R1_SCENE=ecommerce uv run python sft.py \
    --data datasets/ecommerce/train.jsonl --search-backend ecommerce \
    --questions 60 --samples-per-question 4 --min-reward 1.0 --epochs 2 \
    --run-name ecom-sft-coldstart

SEARCH_R1_SCENE=ecommerce uv run python train.py \
    --resume-state '<上一步输出的 state 路径>' \
    --max-steps 20 --search-backend ecommerce --run-name ecom-sft-then-rl
```

拒绝采样的通过率本身就是难度分层。脚本写出的 `rejection_stats.json` 给出每道题的轨迹通过率：

| 通过率 | 含义 | 用途 |
|---|---|---|
| = 0 | base 完全做不出 | SFT 学不到，留给 RL 主攻 |
| ∈ (0, 1) | 有对有错 | SFT 的黄金样本 |
| = 1 | base 已经会 | 对 SFT 和 RL 都几乎无增益（对应单跳题 85.2% → 85.2% 的持平） |

`min-reward ≥ 1.0` 等价于"答案对且引用全对"：答案错时最多只能拿到 0.5 的引用分，达不到 1.0。

### 实测：SFT 的天花板在哪里

60 题 × 4 采样，共 240 条轨迹，通过 109 条（45.4%）。SFT loss 从 0.0237 到 0.0240，两个 epoch 基本没再降。

按题型拆开（`sft_runs/ecom-sft-coldstart/`）：

| 题型 | 题数 | 通过率均值 | hard(=0) | golden(∈(0,1)) | easy(=1) |
|---|---|---|---|---|---|
| 单跳 `ecommerce_single` | 26 | 0.885 | 3 | 0 | 23 |
| 多跳 `ecommerce_multihop` | 31 | 0.105 | 20 | 11 | 0 |
| 政策 `ecommerce_policy` | 3 | 0.333 | 1 | 2 | 0 |
| 合计 | 60 | | 24 | 13 | 23 |

推论：

1. SFT 的增益空间只有 13/60 = 21.7%，能学会的题（golden）一共 13 道。
2. 单跳 88% 的通过率等于 1，SFT 无事可做。这直接解释了评测里单跳 EM 从 85.19% 到 85.19% 没涨：不是训练没跑，是这批题 base 早就饱和了。
3. 多跳 65% 的通过率等于 0，SFT 也学不会，拒绝采样拿不到任何正例。只能靠 RL 用奖励信号去探索，而 RL 确实把多跳从 15.38% 拉到 89.74%。

一句话：SFT 能解决的（easy 23 题）base 早就解决了，base 解决不了的（hard 24 题）SFT 也解决不了。三段式的价值不在 SFT 涨了多少，而在 13/60 这个数字说明了 SFT 已经到顶，后面的增益必须由 RL 提供。
