# Search-R1 Ecommerce Agent

基于 GRPO 强化学习的搜索智能体：**Search-R1 复现 + 中文电商客服场景扩展**。

> **基于 [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)（Apache-2.0）中的 03-search-r1 实验。**
> 本人完成了原实验的完整复现（Qwen3.5-4B + GRPO，PyTRIO 远端训练），并在此基础上扩展了中文电商客服场景：自建电商数据集、本地 Qdrant 向量检索后端、引用可溯源的复合奖励设计。

## 结果速览

### Wikipedia 场景（复现原实验，Wikipedia 免费后端，70 题固定评测集）

| 指标 | Base Model | RL Step-20 | 变化 |
|---|---|---|---|
| EM（宏平均） | 27.1% | **31.4%** | +4.3pp |
| HotpotQA 子集 EM | 30.0% | **50.0%** | +20pp |
| 2WikiMultiHopQA EM | 0.0% | **10.0%** | +10pp |
| 格式合法率 | 50.0% | **70.0%** | +20pp |
| 平均搜索次数 | 2.93 | 2.59 | 搜索更高效 |

### 电商客服场景（本仓库扩展，本地 Qdrant 检索，70 题固定评测集）

<!-- ECOM_EVAL_RESULTS -->

## 这个项目做了什么

### 1. Search-R1 复现

用 PyTRIO 对 Qwen3.5-4B 做 20 步 GRPO 训练：模型在多轮对话中自主决定**何时搜索、搜什么、何时给出最终答案**，以答案 Exact Match 作为奖励信号。训练后模型在多跳问答（HotpotQA）上提升显著。

### 2. 中文电商客服场景扩展（核心增量）

原实验只有英文 Wikipedia 问答。本仓库将其改造为中文电商客服话术场景，解决的是真实客服 Agent 的两大痛点：**回答要有据可查（引用可溯源）、不能编造引用（防幻觉）**。

**数据构建**（`ecommerce/build_data.py`）
- 36 篇虚构品牌的商品文档（型号、价格、保修、物流政策等）——虚构品牌防止模型靠参数化知识"背答案"，逼它走检索
- 204 条训练 + 70 条评测问答对，每题标注 `gold_docs`（答案依据文档编号）

**本地向量检索后端**（`search.py` 新增 `ecommerce` 后端）
- Qdrant 嵌入式存储 + `BGE-small-zh-v1.5` 中文向量模型，免外部搜索 API
- 检索结果回显文档编号（`P001`/`S012` 等 Doc-ID），供模型引用

**引用可溯源的复合奖励**（`reward.py`）

$$r = 1.0 \cdot \text{EM} + 0.1 \cdot \mathbb{1}[\text{有引用}] + 0.2 \cdot \mathbb{1}[\text{引用在真实检索结果中}] + 0.2 \cdot \mathbb{1}[\text{引用命中 gold docs}] - 0.05 \cdot \min(\text{搜索次数}, 4)$$

- **引用真实检索校验**：模型输出的 `Source: P003` 必须真的出现在本轮轨迹检索到的文档里，防止编造引用编号
- **gold 命中奖励**：引用的文档确实是答案依据，鼓励模型引用对的来源而不只是随便引一个
- **搜索成本惩罚**：每次搜索 -0.05，抑制无意义刷搜索

**中文电商客服双协议**（`protocol.py`，环境变量 `SEARCH_R1_SCENE=ecommerce` 切换）
- 用户模拟协议：中文电商咨询提问风格
- 系统协议：要求先检索再回答，答案格式为 `Answer:` + `Source:` 两行

## 目录结构

```
├── train.py            # GRPO 训练（PyTRIO 远端，本地只做数据组装）
├── eval.py             # 统一评测器：base / checkpoint 共用，支持电商指标
├── rollout.py          # 多轮 rollout：搜索工具调用 + 轨迹记录（含 retrieved_doc_ids）★
├── search.py           # 搜索后端：deepseek / wikipedia / zhihu / ecommerce(Qdrant) ★
├── protocol.py         # 双场景协议：英文 Wiki 问答 / 中文电商客服 ★
├── reward.py           # 奖励：原版 EM/Format + 电商引用可溯源复合奖励 ★
├── data.py             # 数据加载（含 gold_docs 字段）★
├── prepare_data.py     # Wikipedia 数据准备（原始 lab 既有）
├── analyse.py          # 评测结果分析（原始 lab 既有）
├── ecommerce/
│   ├── build_data.py   # 电商文档库 + 训练/评测集生成 ★
│   └── index_qdrant.py # BGE 向量化入库本地 Qdrant ★
├── datasets/
│   └── ecommerce/      # 电商文档与训练/评测数据（train 204 / dev 70）
├── eval_result/        # 全部评测原始记录（JSONL，可复查逐题结果）
└── docs/               # 原 lab 教程文档与配图（路径以本仓库根目录为准）
```

标 ★ 的文件相对原仓库做了场景扩展修改；未标注的为原仓库代码。

## 快速开始

### 环境

```bash
# 需要 Python >= 3.13 + uv
uv sync

# PyTRIO 账号（远端训练与采样）：https://pytrio.com
# 首次使用：pytrio login（密钥保存在 ~/.pytrio/config.toml）

# 搜索后端密钥（仅 deepseek/zhihu 后端需要，wiki/ecommerce 后端免费）
cp .env.example .env
```

### 电商场景（本仓库扩展）

```bash
# 1.（可选）重建数据集——仓库已附带 datasets/ecommerce/，可跳过
uv run python ecommerce/build_data.py
# 2. 建向量库（首次运行自动下载 BGE 模型，约 90MB）
uv run python ecommerce/index_qdrant.py

# 3. 训练：20 步 GRPO
SEARCH_R1_SCENE=ecommerce uv run python train.py \
    --max-steps 20 --save-every 5 \
    --data datasets/ecommerce/train.jsonl \
    --search-backend ecommerce \
    --run-name ecom-r1-20step

# 4. 评测：Base vs Step-20 各跑一次 70 题
SEARCH_R1_SCENE=ecommerce \
SEARCH_R1_EVAL_DATA=$PWD/datasets/ecommerce/dev.jsonl \
uv run python eval.py --search-backend ecommerce \
    --output eval_result/ecom_eval_base.jsonl

SEARCH_R1_SCENE=ecommerce \
SEARCH_R1_EVAL_DATA=$PWD/datasets/ecommerce/dev.jsonl \
uv run python eval.py --search-backend ecommerce \
    --model-path 'trio://<你的-step-20-weights-路径>' \
    --output eval_result/ecom_eval_step20.jsonl
```

### Wikipedia 场景（复现原实验）

```bash
# 1. 准备数据（下载 wiki dump 并生成 train/dev/test，详见 docs/lab-readme.md）
uv run python prepare_data.py

# 2. 训练（免费 wikipedia 后端）
uv run python train.py --max-steps 20 --save-every 5 \
    --search-backend wikipedia --run-name search-r1-wiki

# 3. 评测
SEARCH_R1_EVAL_DATA=$PWD/datasets/dev.jsonl \
uv run python eval.py --search-backend wikipedia \
    --output eval_result/eval_results_base_wiki.jsonl
```

> 注：`docs/` 内是原 lab 的教程文档，其中命令以原仓库根目录运行（带 `03-search-r1/` 前缀）；在本仓库中请去掉该前缀、从仓库根目录运行。

## 评测指标说明

| 指标 | 含义 |
|---|---|
| `em/macro` | 各数据源 Exact Match 的宏平均 |
| `format/rate` | 输出格式合法率（电商场景 = Answer + Source 两行齐全） |
| `citation/has_source` | 给出引用行的比例 |
| `citation/source_retrieved` | 引用编号真实出现在本轮检索结果中的比例（防编造引用） |
| `citation/source_gold` | 引用编号命中 gold docs 的比例（引对了来源） |
| `rollout/search_calls` | 平均搜索次数（效率） |

## 训练曲线

- 电商场景：[SwanLab - ecom-r1-20step](https://swanlab.cn/@rickena/agentic-rl-lab-search-r1)
- Wiki 复现：同项目下 `search-r1-baseline-2` 等 run

## 致谢与许可

- 原仓库：[KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) —— 实验框架、训练/评测流程与其教程（见 `docs/`）
- 训练基建：[PyTRIO](https://pytrio.com)；实验跟踪：[SwanLab](https://swanlab.cn)；向量检索：[Qdrant](https://qdrant.tech) + [BGE](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- 许可：Apache-2.0（沿用原仓库）
