# 检索器 top-k 消融

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
