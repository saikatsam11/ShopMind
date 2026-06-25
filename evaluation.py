"""
Phase 4c — Evaluation Metrics
Compares dense-only vs hybrid retrieval on 15 test queries.
Metrics: Precision@3, MRR@3, Average Latency

Note: Relevance is judged by keyword matching on product titles — a proxy
for human labels, standard practice when ground-truth annotations are unavailable.
"""

import importlib.util
import logging
import time
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, Prefetch,
    FusionQuery, Fusion, Range, SparseVector,
)

logging.basicConfig(level=logging.WARNING)   # suppress info logs during eval

# ── Import from Phase 3 ───────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "rag_pipeline", Path(__file__).parent / "03_rag_pipeline.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_embedder   = _mod.load_embedder
QUERY_PREFIX    = _mod.QUERY_PREFIX
QDRANT_HOST     = _mod.QDRANT_HOST
QDRANT_PORT     = _mod.QDRANT_PORT

DENSE_COLLECTION  = "products"
HYBRID_COLLECTION = "products_hybrid"
TOP_K             = 3
BM25_MODEL        = "Qdrant/bm25"

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    {"query": "wireless headphones for gym workout",        "keywords": ["headphone", "earphone", "earbud"]},
    {"query": "laptop for video editing",                   "keywords": ["laptop", "notebook", "macbook"]},
    {"query": "DSLR camera for beginners",                  "keywords": ["camera", "dslr", "mirrorless"]},
    {"query": "mechanical keyboard for programming",        "keywords": ["keyboard"]},
    {"query": "gaming mouse under $50",                     "keywords": ["mouse"], "max_price": 50},
    {"query": "portable bluetooth speaker",                 "keywords": ["speaker"]},
    {"query": "noise cancelling headphones for travel",     "keywords": ["headphone", "earphone", "earbud"]},
    {"query": "smartwatch for fitness tracking",            "keywords": ["watch", "smartwatch", "fitness", "band"]},
    {"query": "webcam for video conferencing",              "keywords": ["webcam", "camera", "web cam"]},
    {"query": "external SSD 1TB",                          "keywords": ["ssd", "solid state", "hard drive", "storage"]},
    {"query": "budget laptop for college students",         "keywords": ["laptop", "notebook", "chromebook"]},
    {"query": "action camera for outdoor adventures",       "keywords": ["camera", "gopro", "action"]},
    {"query": "wireless gaming headset",                    "keywords": ["headset", "headphone", "gaming"]},
    {"query": "USB hub for MacBook",                        "keywords": ["hub", "usb", "adapter", "dock"]},
    {"query": "4K monitor for graphic design",             "keywords": ["monitor", "display", "screen"]},
]


# ── Relevance judgment ────────────────────────────────────────────────────────
def is_relevant(title: str, keywords: list[str]) -> bool:
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in keywords)


# ── Metrics ───────────────────────────────────────────────────────────────────
def precision_at_k(titles: list[str], keywords: list[str], k: int = 3) -> float:
    hits = sum(1 for t in titles[:k] if is_relevant(t, keywords))
    return hits / k


def mrr_at_k(titles: list[str], keywords: list[str], k: int = 3) -> float:
    for rank, title in enumerate(titles[:k], start=1):
        if is_relevant(title, keywords):
            return 1.0 / rank
    return 0.0


# ── Retrieval functions ───────────────────────────────────────────────────────
def dense_retrieve(query: str, embedder, client: QdrantClient,
                   max_price: float = None) -> tuple[list[str], float]:
    conditions = []
    if max_price:
        conditions.append(FieldCondition(key="price_numeric", range=Range(gte=0.01, lte=max_price)))

    vec = embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    t0 = time.time()
    results = client.query_points(
        collection_name=DENSE_COLLECTION,
        query=vec,
        query_filter=Filter(must=conditions) if conditions else None,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )
    latency = time.time() - t0

    titles = [p.payload.get("title", "") for p in results.points]
    return titles, latency


def hybrid_retrieve(query: str, embedder, bm25_model: SparseTextEmbedding,
                    client: QdrantClient, max_price: float = None) -> tuple[list[str], float]:
    conditions = []
    if max_price:
        conditions.append(FieldCondition(key="price_numeric", range=Range(gte=0.01, lte=max_price)))

    dense_vec  = embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    sparse_emb = list(bm25_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    query_filter = Filter(must=conditions) if conditions else None

    t0 = time.time()
    results = client.query_points(
        collection_name=HYBRID_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec,   using="dense",  limit=20),
            Prefetch(query=sparse_vec,  using="sparse", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )
    latency = time.time() - t0

    titles = [p.payload.get("title", "") for p in results.points]
    return titles, latency


# ── Main evaluation ───────────────────────────────────────────────────────────
def main():
    print("Loading models...")
    embedder = load_embedder()
    bm25     = SparseTextEmbedding(model_name=BM25_MODEL)
    client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    dense_p3, dense_mrr, dense_lat   = [], [], []
    hybrid_p3, hybrid_mrr, hybrid_lat = [], [], []

    header = f"{'Query':<45} {'D P@3':>6} {'H P@3':>6} {'D MRR':>6} {'H MRR':>6} {'D ms':>7} {'H ms':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for item in TEST_QUERIES:
        query    = item["query"]
        keywords = item["keywords"]
        price    = item.get("max_price")

        d_titles, d_lat = dense_retrieve(query, embedder, client, price)
        h_titles, h_lat = hybrid_retrieve(query, embedder, bm25, client, price)

        dp3  = precision_at_k(d_titles, keywords)
        hp3  = precision_at_k(h_titles, keywords)
        dmrr = mrr_at_k(d_titles, keywords)
        hmrr = mrr_at_k(h_titles, keywords)

        dense_p3.append(dp3);   hybrid_p3.append(hp3)
        dense_mrr.append(dmrr); hybrid_mrr.append(hmrr)
        dense_lat.append(d_lat * 1000); hybrid_lat.append(h_lat * 1000)

        winner = "H" if hp3 > dp3 else ("D" if dp3 > hp3 else "=")
        q_short = query[:44]
        print(
            f"{q_short:<45} {dp3:>6.2f} {hp3:>6.2f} "
            f"{dmrr:>6.2f} {hmrr:>6.2f} "
            f"{d_lat*1000:>6.0f}ms {h_lat*1000:>6.0f}ms  [{winner}]"
        )

    print("=" * len(header))
    n = len(TEST_QUERIES)
    avg_dp3  = sum(dense_p3) / n;    avg_hp3  = sum(hybrid_p3) / n
    avg_dmrr = sum(dense_mrr) / n;   avg_hmrr = sum(hybrid_mrr) / n
    avg_dlat = sum(dense_lat) / n;   avg_hlat = sum(hybrid_lat) / n

    print(f"{'AVERAGE':<45} {avg_dp3:>6.2f} {avg_hp3:>6.2f} "
          f"{avg_dmrr:>6.2f} {avg_hmrr:>6.2f} "
          f"{avg_dlat:>6.0f}ms {avg_hlat:>6.0f}ms")
    print("=" * len(header))

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Precision@3 : Dense {avg_dp3:.2f}  |  Hybrid {avg_hp3:.2f}  →  "
          + ("Hybrid wins ✅" if avg_hp3 > avg_dp3 else "Dense wins" if avg_dp3 > avg_hp3 else "Tie"))
    print(f"  MRR@3       : Dense {avg_dmrr:.2f}  |  Hybrid {avg_hmrr:.2f}  →  "
          + ("Hybrid wins ✅" if avg_hmrr > avg_dmrr else "Dense wins" if avg_dmrr > avg_hmrr else "Tie"))
    print(f"  Avg Latency : Dense {avg_dlat:.0f}ms  |  Hybrid {avg_hlat:.0f}ms")
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
