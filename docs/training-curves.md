# 训练曲线

图表由 [`plot_curves.py`](plot_curves.py) 从训练日志提取的逐步指标重绘，数据与 SwanLab 云端同源。

### 电商场景（ecom-r1-20step-4）

![电商场景 GRPO 训练曲线](docs/images/curves-ecom.png)

正确率在第 2 步短暂触底 9.4%（首次策略更新后的格式震荡）后快速爬升，第 9 步起稳定在 85%–100%，第 20 步收敛到 100%。奖励收敛到约 1.43，接近复合奖励的理论上限（1.0 EM + 0.5 引用项 − 搜索成本）。搜索次数稳定在 1.4 次/题附近，搜索成本惩罚生效。有 2 个 step 的 loss 显示为跳过，那两步全组零方差（全对或全错），被 Dynamic Sampling 机制跳过，不产生梯度。

### Wikipedia 基线（search-r1-baseline-2）

![Wikipedia 基线 GRPO 训练曲线](docs/images/curves-baseline.png)

训练信号方向正确（奖励 0.03 → 0.5+），但组级方差大：每步只有 64 条 rollout，Wikipedia 多跳任务对 4B 模型偏难，逐步正确率在 6%–58% 之间震荡。所以最终成绩以 70 题固定评测集的双评测为准（EM 27.1% → 31.4%），而不是训练期滚动指标。

SwanLab 在线曲线：

- 电商 run：[ecom-r1-20step-4](https://swanlab.cn/@rickena/agentic-rl-lab-search-r1/runs/cze9su7x)
- Wiki 基线 run：[search-r1-baseline-2](https://swanlab.cn/@rickena/agentic-rl-lab-search-r1/runs/pjirsjub)
