"""检索召回率诊断：量化「top-k 太小」到底是不是评测残留错题的瓶颈。

## 为什么需要这个脚本

评测里有两个数字训练前后完全不动：`em/single` 85.19% → 85.19%、`citation/source_gold`
92.86% → 92.86%。把 5 道残留错题的 `retrieved_doc_ids` 和 `gold_docs` 对起来查，发现
**5/5 全是「该看的文档压根没被检索出来」**——每次只返回 3 篇，目标文档排不进 top-3。

36 篇文档结构完全同构（每篇都是「售价：XXXX元 / 电池容量：…」），「售价」这个词在每一篇
都出现，bge-small-zh-v1.5 的向量检索区分不开。

本脚本用**问题原文当 query**，扫一遍 top-k ∈ {1,3,5,8,16}，看 gold_docs 的召回率怎么变。
如果 k=3 → k=8 召回率显著上升，就说明**瓶颈在检索器配置，而不是策略**——奖励给得再高，
也换不出一个「根本没被检索到」的文档。

## 用法

    uv run python ecommerce/retrieval_recall.py
    uv run python ecommerce/retrieval_recall.py --top-k 1 3 5 8 16 32 --split dev
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
COLLECTION = "ecommerce_docs"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
STORE_PATH = HERE / "qdrant_store"
CACHE_DIR = REPO_ROOT.parent / ".cache" / "fastembed"

# 评测里「答对但引用错 / 引用错」的 5 道残留错题（2026-10-10 从 ecom_eval_step20.jsonl 定位）
KNOWN_FAILURES = ["ecom_0065", "ecom_0093", "ecom_0106", "ecom_0050", "ecom_0226"]


def load_split(split: str) -> list[dict]:
    path = REPO_ROOT / "datasets" / "ecommerce" / f"{split}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> tuple[bool, bool]:
    """返回 (是否召回全部 gold, 是否至少召回一篇 gold)。无 gold 标注时返回 (True, True)。

    无 gold 的题不应计入召回率分母，调用方需先过滤。
    """
    top = retrieved[:k]
    if not gold:
        return True, True
    return all(g in top for g in gold), any(g in top for g in gold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", help="评测划分（dev / test）")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 8, 16, 32])
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="一次查询取回的最大条数；默认取 --top-k 的最大值",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0 表示全部")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_k = args.max_k or max(args.top_k)

    examples = load_split(args.split)
    if args.limit:
        examples = examples[: args.limit]
    labeled = [e for e in examples if e.get("gold_docs")]

    embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(CACHE_DIR))
    client = QdrantClient(path=str(STORE_PATH))

    # 一次取回 max_k 条，再在各档 k 上截断 —— 保证各档看到的排序完全一致
    per_question: dict[str, dict] = {}
    for example in labeled:
        qvec = list(embedder.query_embed(str(example["question"])))[0]
        points = client.query_points(
            collection_name=COLLECTION, query=qvec, limit=max_k
        ).points
        per_question[example["id"]] = {
            "example": example,
            "retrieved": [str(p.payload["doc_id"]) for p in points],
        }

    print(f"split={args.split}  带 gold_docs 的题数={len(labeled)}  max_k={max_k}\n")

    print("=== 总体召回率（问题原文当 query）===")
    print(f"{'k':>4}{'全部命中 gold':>16}{'至少命中一篇':>16}")
    for k in args.top_k:
        full = sum(1 for v in per_question.values() if hit_at_k(v["retrieved"], v["example"]["gold_docs"], k)[0])
        any_ = sum(1 for v in per_question.values() if hit_at_k(v["retrieved"], v["example"]["gold_docs"], k)[1])
        print(f"{k:>4}{full / len(labeled):>15.1%}{any_ / len(labeled):>16.1%}")

    print("\n=== 分题型（全部命中 gold）===")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for v in per_question.values():
        by_source[v["example"].get("data_source", "unknown")].append(v)
    header = f"{'题型':<24}{'n':>4}" + "".join(f"{'k=' + str(k):>9}" for k in args.top_k)
    print(header)
    for source, items in sorted(by_source.items()):
        cells = []
        for k in args.top_k:
            ok = sum(1 for v in items if hit_at_k(v["retrieved"], v["example"]["gold_docs"], k)[0])
            cells.append(f"{ok / len(items):>8.1%}")
        print(f"{source:<24}{len(items):>4}" + "".join(cells))

    print("\n=== 已知 5 道残留错题：gold 文档在各档 k 下的排序位置 ===")
    print(f"{'题号':<12}{'gold_docs':<18}" + "".join(f"{'k=' + str(k):>8}" for k in args.top_k) + "  实际检索期次")
    for qid in KNOWN_FAILURES:
        v = per_question.get(qid)
        if v is None:
            print(f"{qid:<12}(不在本 split)")
            continue
        gold = v["example"]["gold_docs"]
        positions = []
        for g in gold:
            pos = v["retrieved"].index(g) + 1 if g in v["retrieved"] else None
            positions.append(str(pos) if pos else "未召回")
        cells = []
        for k in args.top_k:
            ok = hit_at_k(v["retrieved"], gold, k)[0]
            cells.append(f"{'✓' if ok else '✗':>8}")
        print(f"{qid:<12}{','.join(gold):<18}" + "".join(cells) + f"  rank={positions}")

    print("\n=== 结论 ===")
    k3 = sum(1 for v in per_question.values() if hit_at_k(v["retrieved"], v["example"]["gold_docs"], 3)[0]) / len(labeled)
    k8 = sum(1 for v in per_question.values() if hit_at_k(v["retrieved"], v["example"]["gold_docs"], 8)[0]) / len(labeled)
    print(f"  top-3 全部命中率 {k3:.1%} → top-8 {k8:.1%}（Δ={k8 - k3:+.1%}）")
    if k8 - k3 > 0.05:
        print("  ⇒ gold 文档确实排不进 top-3，但 top-8 能召回：**瓶颈在检索器 top-k，不在策略**。")
        print("     下一步可做：把 search.py 的 top_k 提到 8（或加 BM25 混合 / rerank）后重跑那几题。")
    else:
        print("  ⇒ top-k 不是主因，需要换更强的 embedding 或加混合检索。")


if __name__ == "__main__":
    main()
