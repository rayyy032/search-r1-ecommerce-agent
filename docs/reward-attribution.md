# 奖励项归因

电商场景的复合奖励有 4 个正分项：

$$r = 1.0 \cdot \text{EM} + 0.1 \cdot \text{有引用} + 0.2 \cdot \text{引用真在本轮检索} + 0.2 \cdot \text{引用命中 gold} - 0.05 \cdot \min(\text{搜索次数}, 4)$$

引用命中率不涨，自然的怀疑是"那 0.2 权重太弱"。真实原因不是权重弱。`reward_ablation.py` 对 `eval_result/` 里 70 条落盘轨迹做反事实重打分（把每一项权重单独置 0，看奖励景观怎么变），判据是 GRPO 的组内中心化：

```python
advantage_i = reward_i - mean(reward_group)   # rollout.assign_group_advantages
```

任何在同一组内恒为常数的分项，对 advantage 的贡献恒为 0。于是"把某项权重置 0 后，70 题两两之间的相对顺序有没有变"就能回答"这项到底有没有在产生梯度"。

| 预设（置 0 的项） | 奖励数值变动题数 | 题对顺序变动数 | 判定 |
|---|---|---|---|
| `no_has_source`（−0.1） | 70 / 70 | 0 | 仅全局常数偏移，组内零梯度 |
| `no_source_retrieved`（−0.2） | 70 / 70 | 0 | 仅全局常数偏移，组内零梯度 |
| `no_source_gold`（−0.2） | 65 / 70 | 49 | 有信号，但只来自 5 道题 |
| `no_search_penalty` | 70 / 70 | 927 | 真实起作用（约束检索成本） |
| `em_only` | 70 / 70 | 1004 | 真实起作用 |

跨 checkpoint 的分项触发率：

| 分项 | Base | Step-20 | Δ |
|---|---|---|---|
| EM | 42.86% | 88.57% | +45.71% |
| 有引用 | 98.57% | 100.00% | +1.43% |
| 引用真实检索 | 98.57% | 100.00% | +1.43% |
| 引用命中 gold | 92.86% | 92.86% | +0.00% |
| 搜索惩罚 | 100.00% | 100.00% | +0.00% |

EM 与引用命中 gold 的列联表（Step-20，n=70）：

| | 引用对 | 引用错 |
|---|---|---|
| EM=1 | 61 | 1 |
| EM=0 | 4 | 4 |

Base 阶段是 EM=1 且引用错 0 题，EM=0 但引用对 35 题。所以引用项唯一真正惩罚到的"答对却引错"只有 1 道。

三点结论：

1. 0.3 的引用权重（有引用 0.1 + 引用真实检索 0.2）严格不产生梯度。置 0 后每题奖励都变了，但题对顺序一对没变，它只贡献了一个全局常数偏移。这两项作为防编造引用的硬约束仍然该保留，但不该指望它带来学习信号。
2. `source_gold` 卡死的原因是指标饱和，不是权重太弱。它在两个 checkpoint 上触发率完全相同（92.86%），而同期 EM 从 42.86% 涨到 88.57%。于是它与 EM 的 φ 相关系数从 0.240 升到 0.598，从弱相关变成几乎被 EM 决定，独立贡献被吸收掉大半。
3. 可优化空间只剩 1 道题。Step-20 上"答案对但引用错"只有 1 题，base 阶段是 0 题。这项奖励在起点就接近天花板。

这条负结果和本工作另一个项目（从零手写 GRPO/DAPO 的消融诊断）里 `clip_fraction ≡ 0` 是同一族问题的两种形态：一个是机制没触发，一个是指标已饱和。共同点是配置里打开的开关不等于实际生效的机制。

复现：`uv run python reward_ablation.py`（纯离线，不需要 GPU 和 PyTRIO，结果落在 `eval_result/reward_ablation.json`）。

### 拿因果结论还需要重训

上面的归因是离线观测。要拿到重训后的因果结论，得把奖励权重做成可切换的：

```bash
for cfg in full no_source_gold no_source_retrieved no_citation em_only; do
  SEARCH_R1_SCENE=ecommerce uv run python train.py \
      --reward-config "$cfg" --max-steps 20 --save-every 5 \
      --data datasets/ecommerce/train.jsonl --search-backend ecommerce \
      --run-name "abl-$cfg"
done
```

可选项见 `reward.REWARD_PRESETS`。
