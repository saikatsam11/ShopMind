"""
Phase 4b — Hybrid Index
Migrates 'products' (dense only) → 'products_hybrid' (dense + BM25 sparse).
No GPU needed — reads existing dense vectors from Qdrant, computes BM25 on CPU.

pip install fastembed
"""

import logging
from tqdm import tqdm
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, PointStruct, SparseVector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST       = "localhost"
QDRANT_PORT       = 6333
SOURCE_COLLECTION = "products"
HYBRID_COLLECTION = "products_hybrid"
VECTOR_DIM        = 384
SCROLL_BATCH      = 500
BM25_MODEL        = "Qdrant/bm25"


# ── Collection setup ──────────────────────────────────────────────────────────
def setup_hybrid_collection(client: QdrantClient, recreate: bool = False):
    exists = any(c.name == HYBRID_COLLECTION for c in client.get_collections().collections)

    if exists and not recreate:
        count = client.count(HYBRID_COLLECTION).count
        source_count = client.count(SOURCE_COLLECTION).count
        if count >= source_count:
            log.info(f"  '{HYBRID_COLLECTION}' already complete ({count:,} points) — skipping")
            return False   # signal: nothing to do
        log.info(f"  partial migration detected ({count:,}/{source_count:,}) — resuming")
        return True

    if exists:
        client.delete_collection(HYBRID_COLLECTION)
        log.info(f"  deleted existing '{HYBRID_COLLECTION}'")

    client.create_collection(
        collection_name=HYBRID_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    log.info(f"  created '{HYBRID_COLLECTION}' — dense ({VECTOR_DIM}-dim cosine) + sparse (BM25)")
    return True


# ── Migration ─────────────────────────────────────────────────────────────────
def migrate(client: QdrantClient, bm25_model: SparseTextEmbedding):
    source_count = client.count(SOURCE_COLLECTION).count
    hybrid_count = client.count(HYBRID_COLLECTION).count

    if hybrid_count >= source_count:
        log.info("  Already fully migrated — nothing to do")
        return

    log.info(f"  Migrating {source_count - hybrid_count:,} remaining points")
    offset        = None
    total_done    = 0

    with tqdm(total=source_count, initial=hybrid_count, desc="Migrating", unit="pts") as pbar:
        while True:
            records, next_offset = client.scroll(
                collection_name=SOURCE_COLLECTION,
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break

            # Skip points already in hybrid collection (resume support)
            existing_ids = {
                p.id for p in client.retrieve(
                    collection_name=HYBRID_COLLECTION,
                    ids=[r.id for r in records],
                    with_payload=False,
                    with_vectors=False,
                )
            }
            new_records = [r for r in records if r.id not in existing_ids]

            if new_records:
                texts = [
                    r.payload.get("combined_text") or r.payload.get("title", "")
                    for r in new_records
                ]
                sparse_embs = list(bm25_model.embed(texts))

                points = [
                    PointStruct(
                        id=rec.id,
                        vector={
                            "dense": rec.vector,
                            "sparse": SparseVector(
                                indices=sp.indices.tolist(),
                                values=sp.values.tolist(),
                            ),
                        },
                        payload=rec.payload,
                    )
                    for rec, sp in zip(new_records, sparse_embs)
                ]
                client.upsert(collection_name=HYBRID_COLLECTION, points=points)
                pbar.update(len(new_records))
                total_done += len(new_records)

            if next_offset is None:
                break
            offset = next_offset

    log.info(f"  Migration done — {total_done:,} new points added")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)

    log.info("=== Step 1: Setting up hybrid collection ===")
    setup_hybrid_collection(client, recreate=False)

    log.info("=== Step 2: Loading BM25 model ===")
    bm25 = SparseTextEmbedding(model_name=BM25_MODEL)
    log.info(f"  '{BM25_MODEL}' ready")

    log.info("=== Step 3: Migrating dense + adding BM25 sparse vectors ===")
    migrate(client, bm25)

    final_count = client.count(HYBRID_COLLECTION).count
    print("\n" + "=" * 60)
    print(f"Collection : {HYBRID_COLLECTION}")
    print(f"Vectors    : {final_count:,}")
    print(f"Dense      : BGE-small-en-v1.5 (384-dim, cosine)")
    print(f"Sparse     : BM25 (Qdrant/bm25)")
    print(f"Fusion     : RRF (Reciprocal Rank Fusion) at query time")
    print("=" * 60)


if __name__ == "__main__":
    main()
