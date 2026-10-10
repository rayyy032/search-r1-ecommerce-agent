# 代码结构对应 agentic RL 的哪些层

Agent + RL 方向的核心工程件不是一个 prompt 编排层，而是一个多轮可交互、奖励可验证、轨迹可回放的环境。仓库里的组件刚好是这个结构：

| agentic RL 框架的分层 | 本仓库对应文件 | 作用 |
|---|---|---|
| Environment / Tools | `search.py`（`ecommerce` 后端，Qdrant 本地向量检索） | 模型可调用的检索工具，返回带 Doc-ID 的观测 |
| Multi-turn rollout | `rollout.py` | 多轮「生成 → 解析 tool_call → 执行 → 拼接下一轮 prompt」 |
| Verifiable reward | `reward.py` | 不需要奖励模型，答案 EM 和引用可溯源全部可规则判定 |
| Policy optimization | `train.py` | 组内相对优势（GRPO）+ 动态 micro-batch 装箱 |
| Diagnostics | `reward_ablation.py`、`analyse.py` | 奖励项归因、逐题 badcase 复查 |

后续如果要升级，方向是按 RAGEN / verl-agent 的目录约定重整（`environments/`、`memory/`、`multi_turn_rollout/`、`reward_manager/`）。
