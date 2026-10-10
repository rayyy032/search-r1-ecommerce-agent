# 推理部署压测

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
