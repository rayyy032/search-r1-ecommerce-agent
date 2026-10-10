# Ecom Search-R1

一个中文电商客服 Agent。用户问一个问题（比如"XX 牌空气炸锅保修多久，多少钱"），模型先判断要不要查自家商品文档，查完再回答，并且答案末尾必须给出出处编号。

商品文档里的品牌全是虚构的，模型在预训练里没见过，所以答对只能靠检索。用 GRPO 训练 Qwen3.5-4B 之后，70 道题的 EM 从 41.9% 涨到 91.6%。

![架构](docs/images/architecture.svg)

## 快速开始

```bash
# 需要 Python >= 3.13 和 uv
uv sync

# 建本地向量库（首次会自动下载 BGE 模型，约 90MB）
uv run python ecommerce/index_qdrant.py

# 跑自检，不需要 GPU
uv run --extra dev pytest tests/ -q

# 20 步 GRPO 训练（GPU 在远端 PyTRIO）
SEARCH_R1_SCENE=ecommerce uv run python train.py \
    --max-steps 20 --save-every 5 \
    --data datasets/ecommerce/train.jsonl \
    --search-backend ecommerce \
    --run-name ecom-r1-20step

# 评测：base 与 checkpoint 各跑 70 题
SEARCH_R1_SCENE=ecommerce SEARCH_R1_EVAL_DATA=$PWD/datasets/ecommerce/dev.jsonl \
uv run python eval.py --search-backend ecommerce --output eval_result/ecom_eval_base.jsonl
```

训练和采样走远端 PyTRIO，需要一个 API key（`pytrio login`，密钥存在 `~/.pytrio/config.toml`）。本机只做数据组装和本地检索，不需要 GPU。

**唯一需要 GPU 的脚本是 `bench_serving.py`**（推理部署压测），用仓库里的 `kaggle_bench_serving.ipynb` 在 Kaggle T4 上跑。本地是 macOS，没有 CUDA。

## 我做了什么

基于 [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)（Apache-2.0）的 03-search-r1 实验框架扩展。相对原仓库新增或修改的文件在目录结构里用 ★ 标出，其余是原仓库代码。

分工上，原仓库给的是英文 Wikipedia 问答的骨架，我做的是：

- 把场景换成中文电商客服，自己生成 36 篇商品文档和 204/70 的训练评测集（`ecommerce/build_data.py`，seed=42 纯代码生成，每题带 `gold_docs` 标注答案依据）
- 接了本地检索环境：Qdrant 嵌入式向量库 + `BGE-small-zh-v1.5`，检索结果带 Doc-ID 回显，供模型引用
- 设计了引用可溯源的复合奖励，并且用反事实重打分量化每个奖励分项实际产生了多少梯度（`reward.py`、`reward_ablation.py`）
- 补了 SFT 冷启动环节（拒绝采样），把 base → SFT → RL 的三段式补齐（`sft.py`）
- 跑了检索器 top-k 消融，定位残留错题的瓶颈（`ecommerce/retrieval_recall.py`）
- 做了推理部署压测（`bench_serving.py`）

## 结果

电商场景，本地 Qdrant 检索，70 题固定评测集：

| 指标 | Base | RL Step-20 | 变化 |
|---|---|---|---|
| EM（宏平均） | 41.9% | 91.6% | +49.8pp |
| 多跳题 EM（需 ≥2 次检索） | 15.4% | 89.7% | +74.4pp |
| 政策类题 EM（保修/物流/退换） | 25.0% | 100.0% | +75.0pp |
| 单跳题 EM | 85.2% | 85.2% | 持平 |
| 格式合法率（Answer + Source 两行齐全） | 98.6% | 100.0% | +1.4pp |
| 引用存在率 | 98.6% | 100.0% | +1.4pp |
| 引用真实检索率（防编造） | 98.6% | 100.0% | +1.4pp |
| 引用命中 gold | 92.9% | 92.9% | 持平 |
| 平均搜索次数 | 1.44 | 1.56 | |

涨的部分符合预期：多跳题要先查商品再查政策，RL 之后模型学会了多轮检索再作答，引用编号全部来自本轮检索结果，没有编造。

**有两个数字训练前后完全不动**，这比涨点更值得说：

1. 单跳题 85.19% → 85.19%。一开始的解释是"base 已经会的题 RL 不碰"，后来发现这个解释不对。真实瓶颈在检索器的 top-k，把 k 从 3 提到 8、模型权重完全不动，单跳题就从 85.19% 变成 100%。
2. 引用命中 gold 卡在 92.86%。既不是噪声也不是权重太弱，奖励归因显示它与 EM 的排序几乎完全一致（φ 从 0.240 升到 0.598），可优化空间只剩 1 道题；top-k 消融说明另一半原因同样是召回。

## 检索器 top-k 消融

把残留错题的 `retrieved_doc_ids` 和 `gold_docs` 对起来看，5 道错题 5 道都是该看的文档压根没被检索出来，而线上每次只返回 3 篇。36 篇文档结构完全同构（每篇都有"售价：…"），`bge-small-zh-v1.5` 区分不开，目标文档排不进 top-3。

先用问题原文当 query 扫 top-k，看 gold 有没有被召回（纯离线，不需要模型，`ecommerce/retrieval_recall.py`）：

| k | 全部命中 gold | 至少命中一篇 |
|---|---|---|
| 1 | 32.9% | 71.4% |
| 3（线上配置） | 60.0% | 87.1% |
| 5 | 74.3% | 95.7% |
| 8 | 90.0% | 100% |
| 16 | 100% | 100% |

5 道已知错题的 gold 排序位置：`ecom_0065→6`、`ecom_0093→3`、`ecom_0106→5`、`ecom_0050→5`、`ecom_0226→8,9`。

然后做单变量对照，只改 `top_k`，模型完全不动：

| 指标 | base · top_k=3 | base · top_k=8 | Δ |
|---|---|---|---|
| EM（宏平均） | 41.86% | 54.49% | +12.63pp |
| 单跳 `em/single` | 85.19% | 100.0% | +14.81pp |
| 多跳 `em/multihop` | 15.38% | 38.46% | +23.08pp |
| 政策 `em/policy` | 25.0% | 25.0% | 0 |
| `citation/source_gold` | 92.86% | 100.0% | +7.14pp |
| `format/rate` | 98.57% | 100.0% | +1.43pp |
| `search/results` | 3.0 | 8.0 | 配置已生效 |

三点结论：

1. 诊断成立。单跳 85.19% → 100% 和 `source_gold` 92.86% → 100% 是同一件事，都是 gold 文档排不进 top-3。奖励给得再高，也换不出一个根本没被检索到的文档。
2. 政策题不涨是必要的反例。政策类只有 4 道，离线实验显示 k=3 时召回率已经 100%，所以它的问题是推理不是检索。修复只对"召回不足"的题有效，说明这个诊断有边界，不是一个万能解释。
3. 原始 baseline 被低估了。`41.86% → 91.64%` 的 +49.8pp 里有 12.63pp 来自检索器配置，不是 RL。检索器统一到 top_k=8 之后，RL 的真实增益才可比。

复现：

```bash
SEARCH_R1_SCENE=ecommerce SEARCH_R1_EVAL_DATA=$PWD/datasets/ecommerce/dev.jsonl \
uv run python eval.py --search-backend ecommerce --top-k 8 \
    --output eval_result/ecom_eval_base_topk8.jsonl
```

## 奖励项归因

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

## SFT 冷启动

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

## 推理部署压测

这一节对应 JD 里提及率 37.2% 的"部署"要求（Agent 岗 49%），三个项目里唯一全空的一块。写在 Kaggle T4 上，脚本是 `bench_serving.py`。

HF `generate` 批量基线，模型 Qwen3.5-4B，`max_new_tokens=128`（左填充 + 显式 fp16）：

| 批大小 | 吞吐 tok/s | 生成墙钟 s | 延迟 p95 s |
|---|---|---|---|
| 1 | 10.6 | 12.0 | 12.04 |
| 2 | 8.6 | 29.6 | 14.81 |
| 4 | 15.9 | 32.3 | 8.07 |
| 8 | 27.4 | 37.2 | 4.65 |

批大小 1→2 吞吐反而降了 19%：朴素批量生成要把 prompt padding 到批内最长序列，小 batch 的并行收益抵不过 padding 的无效计算，到 batch 8 才拿到 2.6× 加速。

早期两轮数字（9.6/9.0/17.0/30.5 与 9.2/8.8/16.6/29.3 tok/s）作废的原因：当时用 `AutoTokenizer` 默认的右填充做批量生成，而 decoder-only 模型必须左填充，否则模型会从 pad token 之后续写；同时 T4 不支持 bf16，`dtype="auto"` 会加载 bf16 并退化成慢速实现。这两处已修，上表是修正后重跑的结果。

HF 的 `generate` 没有连续批处理，只能整批前向，所以这里的 `concurrency` 实际是 batch size，p95 是"整批墙钟 ÷ 请求数"的摊销值，不是单请求延迟。这一点自己要先讲清楚，否则会被面试官纠正。

**vLLM 两个后端在这台 Kaggle 机器上都没跑通**，原因不是脚本：

- Kaggle 镜像的 CUDA runtime 是 12.8，而 `pip install vllm` 拉到的轮子编译在 CUDA 13 上，`import vllm` 直接报 `ImportError: libcudart.so.13: cannot open shared object file`。
- 它还会把 `torch` 一并换成 cu130，这一步之后连原本跑通的 HF 基线都崩了（新进程里 torch 是 13.0、torchaudio 还是 12.8）。
- notebook 的 Cell 5 做过 torchaudio 对齐，只能解决"torch 与 torchaudio 互不一致"，解决不了"vLLM 要 CUDA 13、镜像只有 CUDA 12.8"。

原则是 vLLM 没跑通就只写 HF 基线的真实数字，外加说明环境原因，不编加速比。真要跑 vLLM 得换成 CUDA 12.8 编译的旧版轮子逐个试版本，每次约 10 分钟额度，换来的只是一个加分数字，这笔账不划算。

## 代码结构对应 agentic RL 的哪些层

Agent + RL 方向的核心工程件不是一个 prompt 编排层，而是一个多轮可交互、奖励可验证、轨迹可回放的环境。仓库里的组件刚好是这个结构：

| agentic RL 框架的分层 | 本仓库对应文件 | 作用 |
|---|---|---|
| Environment / Tools | `search.py`（`ecommerce` 后端，Qdrant 本地向量检索） | 模型可调用的检索工具，返回带 Doc-ID 的观测 |
| Multi-turn rollout | `rollout.py` | 多轮「生成 → 解析 tool_call → 执行 → 拼接下一轮 prompt」 |
| Verifiable reward | `reward.py` | 不需要奖励模型，答案 EM 和引用可溯源全部可规则判定 |
| Policy optimization | `train.py` | 组内相对优势（GRPO）+ 动态 micro-batch 装箱 |
| Diagnostics | `reward_ablation.py`、`analyse.py` | 奖励项归因、逐题 badcase 复查 |

后续如果要升级，方向是按 RAGEN / verl-agent 的目录约定重整（`environments/`、`memory/`、`multi_turn_rollout/`、`reward_manager/`）。

## 仓库结构

```
├── train.py            # GRPO 训练（PyTRIO 远端，本地只做数据组装）★
├── sft.py              # SFT 冷启动：拒绝采样 + assistant-only 交叉熵 ★新增
├── eval.py             # 统一评测器：base / checkpoint 共用，支持电商指标 ★
├── rollout.py          # 多轮 rollout：搜索工具调用 + 轨迹记录（含 retrieved_doc_ids）★
├── search.py           # 搜索后端：deepseek / wikipedia / zhihu / ecommerce(Qdrant) ★
├── protocol.py         # 双场景协议：英文 Wiki 问答 / 中文电商客服 ★
├── reward.py           # 奖励：原版 EM/Format + 电商引用可溯源复合奖励（权重可配置）★
├── reward_ablation.py  # 奖励项归因：反事实重打分，量化每个分项的梯度贡献 ★新增
├── bench_serving.py    # 部署压测：vLLM vs HF generate 的吞吐 / TTFT / 尾延迟 ★新增
├── data.py             # 数据加载（含 gold_docs 字段）★
├── prepare_data.py     # Wikipedia 数据准备（原仓库既有）
├── analyse.py          # 评测结果分析（原仓库既有）
├── plot_curves.py      # 从训练日志重绘训练曲线 ★
├── tests/
│   ├── test_reward.py  # 奖励自检：用落盘的真实轨迹把奖励公式钉死 ★新增
│   └── test_cli.py     # CLI 参数自检：抓「用了但没定义的参数」 ★新增
├── ecommerce/
│   ├── build_data.py       # 电商文档库 + 训练/评测集生成（seed=42 全代码生成）★
│   ├── index_qdrant.py     # BGE 向量化入库本地 Qdrant ★
│   └── retrieval_recall.py # 检索召回上限扫描（纯离线，不需要模型）★新增
├── datasets/ecommerce/ # 电商文档与训练/评测数据（train 204 / dev 70）
├── eval_result/        # 全部评测原始记录 + 奖励归因结果（JSONL / JSON）
└── docs/
    ├── images/         # 架构图、训练曲线
    └── training_data/  # 从训练日志提取的逐步指标（图表数据源，与 SwanLab 同源）
```

## Wikipedia 场景（改造前的管线验证）

在改成电商之前，先在原论文的 Wikipedia 场景完整跑通了一遍，作为"管线没写错"的基线。70 题固定评测集：

| 指标 | Base | RL Step-20 | 变化 |
|---|---|---|---|
| EM（宏平均） | 27.1% | 31.4% | +4.3pp |
| HotpotQA 子集 EM | 30.0% | 50.0% | +20pp |
| 2WikiMultiHopQA EM | 0.0% | 10.0% | +10pp |
| 格式合法率 | 50.0% | 70.0% | +20pp |
| 平均搜索次数 | 2.93 | 2.59 | |

```bash
uv run python prepare_data.py
uv run python train.py --max-steps 20 --save-every 5 \
    --search-backend wikipedia --run-name search-r1-wiki
SEARCH_R1_EVAL_DATA=$PWD/datasets/dev.jsonl \
uv run python eval.py --search-backend wikipedia \
    --output eval_result/eval_results_base_wiki.jsonl
```

`docs/` 里是原仓库的教程文档，其中的命令以原仓库根目录运行（带 `03-search-r1/` 前缀），在本仓库里要去掉该前缀。

## 评测指标

| 指标 | 含义 |
|---|---|
| `em/macro` | 各数据源 Exact Match 的宏平均 |
| `em/{single,policy,multihop}` | 按题型拆分的 EM（单跳 / 政策 / 多跳） |
| `format/rate` | 输出格式合法率（电商场景 = Answer + Source 两行齐全） |
| `citation/has_source` | 给出引用行的比例 |
| `citation/source_retrieved` | 引用编号真实出现在本轮检索结果中的比例（防编造引用） |
| `citation/source_gold` | 引用编号命中 gold docs 的比例 |
| `rollout/search_calls` | 平均搜索次数 |
| `search/*` | 检索成功率 / 超时率 / 429 率 / 平均延迟 |

## 训练曲线

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

## 致谢与许可

- 原仓库：[KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)，实验框架、训练评测流程与教程（见 `docs/`）
- 训练基建：[PyTRIO](https://pytrio.com)；实验跟踪：[SwanLab](https://swanlab.cn)；向量检索：[Qdrant](https://qdrant.tech) + [BGE](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- 许可：Apache-2.0，沿用原仓库
