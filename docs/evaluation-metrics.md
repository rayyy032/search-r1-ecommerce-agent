# 评测指标

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
