"""把训练好的 Qwen3.5-4B 策略服务化，并做并发压测（vLLM vs HF generate）。

## 为什么这个脚本在仓库里

403 条真实 JD 里「部署 / 推理加速」的正文提及率是 **37.2%**（Agent 岗 49%），
而这个仓库原来只有「训练 + 评测」，缺了「把模型服务出去并压测」这一段。
本脚本补上这一段，产出一组简历可用的数字：
**并发 1→16 时吞吐与尾延迟怎么变、vLLM 相对 HF generate 快多少、TTFT 随并发涨多少**。

## 两种模式

- `--mode server`（默认，JD 说的那种部署）：`vllm serve` 起 OpenAI 兼容服务，
  用流式请求测 **TTFT（首 token 延迟）/ 尾延迟 p95 / 吞吐**，并发逐级扫。
  复用仓库已有的 `openai` 依赖，不额外引入 HTTP 客户端。
- `--mode offline`：`vllm.LLM` 离线批量推理，测**吞吐上界**（无 TTFT 概念）。
- `--backend hf`：`transformers.generate` 批量基线，用来量化「部署优化」的收益。

## 用法（需要 GPU，本机无 CUDA 时只能做静态检查）

    # 起 vllm 服务并逐级压测
    uv run python bench_serving.py --mode server --model Qwen/Qwen3.5-4B \
        --concurrency 1 2 4 8 16 --requests-per-level 32

    # HF 基线（同一批 prompt、同一 max_new_tokens）
    uv run python bench_serving.py --backend hf --concurrency 1 2 4 8 \
        --requests-per-level 8

    # 输出写到 bench_result/serving_bench.json，可直接喂给 plot / README
"""

import argparse
import asyncio
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_DATA = REPO_ROOT / "datasets" / "ecommerce" / "dev.jsonl"
HEALTH_TIMEOUT_SECONDS = 600  # 4B 模型首次加载 + 编译 CUDA graph 需要时间
HEALTH_POLL_SECONDS = 3.0


def build_prompts(data_path: Path, limit: int) -> list[list[dict[str, str]]]:
    """从评测集题目拼一批 chat messages，作为压测输入。"""
    if not data_path.exists():
        raise FileNotFoundError(f"找不到 prompt 数据：{data_path}")
    questions: list[str] = []
    with data_path.open(encoding="utf-8") as file:
        for line in file:
            questions.append(json.loads(line)["question"])
    if not questions:
        raise ValueError(f"{data_path} 里没有题目")
    selected = [questions[index % len(questions)] for index in range(limit)]
    return [
        [
            {
                "role": "system",
                "content": (
                    "你是中文电商客服助手。先检索再回答，最终输出恰好两行："
                    "一行 `Answer: ` 加答案，一行 `Source: ` 加文档编号。"
                ),
            },
            {"role": "user", "content": question},
        ]
        for question in selected
    ]


def percentile(values: list[float], ratio: float) -> float:
    """取分位数；样本不足时退化为最大值。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


def summarize(latencies: list[float], ttfts: list[float], completion_tokens: int, wall: float) -> dict:
    """把一轮压测的原始统计量整理成指标字典。"""
    return {
        "requests": len(latencies),
        "wall_seconds": wall,
        "throughput_tokens_per_second": completion_tokens / wall if wall > 0 else 0.0,
        "throughput_requests_per_second": len(latencies) / wall if wall > 0 else 0.0,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "ttft_p50_seconds": percentile(ttfts, 0.50) if ttfts else None,
        "ttft_p95_seconds": percentile(ttfts, 0.95) if ttfts else None,
        "completion_tokens": completion_tokens,
    }


# --------------------------- server 模式（vLLM + OpenAI 兼容接口） ---------------------------


def wait_for_server(base_url: str, process: subprocess.Popen) -> None:
    """轮询 /models 直到服务可用；进程提前退出则直接报错。"""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vllm serve 提前退出，返回码 {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(HEALTH_POLL_SECONDS)
    raise TimeoutError(f"{HEALTH_TIMEOUT_SECONDS}s 内服务未就绪：{base_url}")


async def run_one_request(client, model: str, messages, max_tokens: int) -> tuple[float, float | None, int]:
    """发一条流式请求，返回 (总延迟, 首 token 延迟, 生成 token 数)。"""
    started = time.monotonic()
    ttft: float | None = None
    tokens = 0
    stream = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, stream=True
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.monotonic() - started
            tokens += 1  # 流式 chunk 数作为 token 数的上界估计
    return time.monotonic() - started, ttft, tokens


async def run_concurrency_level(client, model: str, prompts, concurrency: int, max_tokens: int) -> dict:
    """在一个并发档位上跑完所有 prompt，返回该档位的汇总指标。"""
    latencies: list[float] = []
    ttfts: list[float] = []
    completion_tokens = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(messages):
        nonlocal completion_tokens
        async with semaphore:
            latency, ttft, tokens = await run_one_request(client, model, messages, max_tokens)
        latencies.append(latency)
        if ttft is not None:
            ttfts.append(ttft)
        completion_tokens += tokens

    started = time.monotonic()
    await asyncio.gather(*(worker(messages) for messages in prompts))
    return {"concurrency": concurrency, **summarize(latencies, ttfts, completion_tokens, time.monotonic() - started)}


def run_server_mode(args, prompts) -> list[dict]:
    """起 vllm serve 子进程，逐级并发压测，最后关掉服务。"""
    from openai import AsyncOpenAI

    base_url = f"http://127.0.0.1:{args.port}/v1"
    command = [
        "vllm", "serve", args.model,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    print(f"启动服务：{shlex.join(command)}")
    process = subprocess.Popen(command)
    results: list[dict] = []
    try:
        wait_for_server(base_url, process)
        print("服务就绪，开始压测")
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        for concurrency in args.concurrency:
            level_prompts = [prompts[index % len(prompts)] for index in range(args.requests_per_level)]
            # 每个档位先预热，避免把首次 CUDA graph 捕获算进延迟
            for _ in range(args.warmup):
                asyncio.run(run_one_request(client, args.model, level_prompts[0], args.max_new_tokens))
            result = asyncio.run(
                run_concurrency_level(client, args.model, level_prompts, concurrency, args.max_new_tokens)
            )
            results.append(result)
            print(
                f"  并发 {concurrency:>3} | 吞吐 {result['throughput_tokens_per_second']:>8.1f} tok/s"
                f" | TTFT p50 {result['ttft_p50_seconds'] or 0:.3f}s"
                f" | 延迟 p95 {result['latency_p95_seconds']:.3f}s"
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    return results


# --------------------------- offline / HF 模式 ---------------------------


def run_offline_mode(args, prompts) -> list[dict]:
    """用 vllm.LLM 做离线批量推理，测吞吐上界。"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    results: list[dict] = []
    for concurrency in args.concurrency:
        batch = [
            tokenizer.apply_chat_template(prompts[index % len(prompts)], tokenize=False, add_generation_prompt=True)
            for index in range(args.requests_per_level)
        ]
        started = time.monotonic()
        outputs = llm.generate(batch, sampling)
        wall = time.monotonic() - started
        completion_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
        # 离线批处理没有单请求延迟概念，用「整批墙钟/请求数」作为摊销延迟
        amortized = wall / max(len(outputs), 1)
        results.append(
            {
                "concurrency": concurrency,
                **summarize([amortized] * len(outputs), [], completion_tokens, wall),
                "note": "离线批量：无 TTFT，延迟为整批摊销值",
            }
        )
        print(f"  批大小 {concurrency:>3} | 吞吐 {results[-1]['throughput_tokens_per_second']:>8.1f} tok/s | 墙钟 {wall:.2f}s")
    return results


def load_hf_model(model_name: str):
    """加载 HF 权重，返回 (tokenizer, model)。

    Qwen3.5-4B 在 HF 上注册的 pipeline_tag 是 `image-text-to-text`（多模态架构），
    直接 `AutoModelForCausalLM` 会因架构不匹配而报错；这里按「多模态 → 因果 LM」
    的顺序逐级回退，并兼容 transformers 4.x(`torch_dtype`) / 5.x(`dtype`) 两种参数名。
    任何一级成功即返回，全部失败才抛出最后一个错误。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    candidates = []
    try:  # transformers 新版本才有 ImageTextToText 基类
        from transformers import AutoModelForImageTextToText

        candidates.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    candidates.append(AutoModelForCausalLM)
    last_error: Exception | None = None
    for model_cls in candidates:
        for dtype_kwarg in ("dtype", "torch_dtype"):
            try:
                model = model_cls.from_pretrained(
                    model_name, device_map="auto", trust_remote_code=True, **{dtype_kwarg: "auto"}
                )
                return tokenizer, model.eval()
            except Exception as error:  # 记录后在下一组合重试
                last_error = error
    raise RuntimeError(f"加载 {model_name} 失败（已尝试 {[c.__name__ for c in candidates]}）：{last_error}")


def run_hf_mode(args, prompts) -> list[dict]:
    """transformers.generate 批量基线，量化部署优化的收益。"""
    import torch

    tokenizer, model = load_hf_model(args.model)
    results: list[dict] = []
    offset = 0
    for concurrency in args.concurrency:
        batch = [
            tokenizer.apply_chat_template(
                prompts[(offset + index) % len(prompts)],
                tokenize=False,
                add_generation_prompt=True,
            )
            for index in range(concurrency)
        ]
        offset += concurrency
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        wall = time.monotonic() - started
        completion_tokens = int((generated.shape[1] - inputs["input_ids"].shape[1]) * len(batch))
        amortized = wall / len(batch)
        results.append(
            {
                "concurrency": concurrency,
                **summarize([amortized] * len(batch), [], completion_tokens, wall),
                "note": "HF generate 基线：无 TTFT，延迟为整批摊销值",
            }
        )
        print(f"  批大小 {concurrency:>3} | 吞吐 {results[-1]['throughput_tokens_per_second']:>8.1f} tok/s | 墙钟 {wall:.2f}s")
    return results


def parse_args() -> argparse.Namespace:
    """解析压测参数。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--backend", choices=["server", "offline", "hf"], default="server")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prompt-data", type=Path, default=DEFAULT_PROMPT_DATA)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "bench_result" / "serving_bench.json")
    return parser.parse_args()


def main() -> None:
    """跑压测并落盘结果。"""
    args = parse_args()
    if args.backend == "server" and not _has_command("vllm"):
        sys.exit("找不到 vllm 命令；请先 pip install vllm，或改用 --backend offline / hf")
    prompts = build_prompts(args.prompt_data, max(args.requests_per_level, max(args.concurrency)))
    print(f"后端={args.backend}  模型={args.model}  并发档位={args.concurrency}")
    if args.backend == "server":
        results = run_server_mode(args, prompts)
    elif args.backend == "offline":
        results = run_offline_mode(args, prompts)
    else:
        results = run_hf_mode(args, prompts)

    payload = {
        "model": args.model,
        "backend": args.backend,
        "max_new_tokens": args.max_new_tokens,
        "requests_per_level": args.requests_per_level,
        "levels": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")


def _has_command(name: str) -> bool:
    """检查命令是否在 PATH 上。"""
    from shutil import which

    return which(name) is not None


if __name__ == "__main__":
    main()
