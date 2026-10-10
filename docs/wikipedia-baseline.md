# Wikipedia 场景（改造前的管线验证）

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
