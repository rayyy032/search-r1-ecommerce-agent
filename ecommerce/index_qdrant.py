"""把 docs.jsonl 的商品 / 政策文档向量化并写入本地 Qdrant 集合。

- Embedding：BAAI/bge-small-zh-v1.5（512 维中文向量，经 fastembed 本地 ONNX 推理）
- Qdrant：嵌入式文件存储（ecommerce/qdrant_store/），无需 Docker / 外部服务
- 模型缓存重定向到项目内 .cache/fastembed，避免写入用户主目录

用法：uv run python 03-search-r1/ecommerce/index_qdrant.py
"""

import json
import os
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

HERE = Path(__file__).resolve().parent
COLLECTION = "ecommerce_docs"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
STORE_PATH = HERE / "qdrant_store"
CACHE_DIR = HERE.parent.parent / ".cache" / "fastembed"

# 在初始化 fastembed 前设置缓存目录
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(CACHE_DIR))


def load_docs() -> list[dict]:
    path = HERE.parent / "datasets" / "ecommerce" / "docs.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> None:
    docs = load_docs()
    embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(CACHE_DIR))
    vectors = list(embedder.embed([doc["content"] for doc in docs]))

    client = QdrantClient(path=str(STORE_PATH))
    client.delete_collection(collection_name=COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance="Cosine"),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "category": doc.get("category", "政策"),
                    "content": doc["content"],
                },
            )
            for i, (doc, vector) in enumerate(zip(docs, vectors), start=1)
        ],
    )

    # 冒烟验证：中文查询应召回相关文档
    probe = "七天无理由退货的运费谁承担"
    qvec = list(embedder.query_embed(probe))[0]
    hits = client.query_points(collection_name=COLLECTION, query=qvec, limit=3).points
    print(f"入库完成：{len(docs)} 篇文档 -> {STORE_PATH}")
    print(f"验证查询「{probe}」：")
    for hit in hits:
        print(f"  - {hit.payload['doc_id']} {hit.payload['title']} (score={hit.score:.3f})")


if __name__ == "__main__":
    main()
