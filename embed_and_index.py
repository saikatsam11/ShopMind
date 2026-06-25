"""
Phase 2 — Embed + Qdrant Index
Loads products_200k.csv, embeds combined_text with BGE-small on CUDA,
upserts vectors + metadata payload into Qdrant.

Before running:
    docker run -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
"""

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = Path(r"E:\Placement\Rag_Project\products_200k.csv")
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIM      = 384
EMBED_BATCH     = 128     # RTX 3050 4GB safe
UPSERT_BATCH    = 256     # points per Qdrant upsert call
CHUNK_SIZE      = 25_000  # rows to embed per GPU pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_parse(val, default):
    if isinstance(val, (list, dict)):
        return val
    try:
        if pd.isna(val):
            return default
    except TypeError:
        pass
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return default


def _sub_category(cats_val) -> str:
    cats = _safe_parse(cats_val, [])
    return cats[-1] if isinstance(cats, list) and cats else ""


def _brand(details_val) -> str:
    details = _safe_parse(details_val, {})
    if not isinstance(details, dict):
        return ""
    return str(details.get("Manufacturer") or details.get("Brand") or "")


def _safe_float(val):
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ── Qdrant setup ──────────────────────────────────────────────────────────────
def setup_collection(client: QdrantClient, recreate: bool = False):
    exists = any(c.name == COLLECTION_NAME for c in client.get_collections().collections)

    if exists and not recreate:
        count = client.count(COLLECTION_NAME).count
        log.info(f"  collection '{COLLECTION_NAME}' exists — {count:,} points")
        return

    if exists:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"  deleted existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    log.info(f"  created collection '{COLLECTION_NAME}'  dim={VECTOR_DIM}  distance=Cosine")


def create_payload_indexes(client: QdrantClient):
    for field, schema in [
        ("price_numeric",  PayloadSchemaType.FLOAT),
        ("average_rating", PayloadSchemaType.FLOAT),
        ("main_category",  PayloadSchemaType.KEYWORD),
        ("sub_category",   PayloadSchemaType.KEYWORD),
        ("brand",          PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
            log.info(f"  payload index created: {field}")
        except Exception as e:
            log.warning(f"  skipping index '{field}' — {e}")


# ── Embed + upsert ────────────────────────────────────────────────────────────
def embed_and_upsert(df: pd.DataFrame, model: SentenceTransformer, client: QdrantClient, device: str):
    texts = df["combined_text"].tolist()
    total = len(df)

    # Resume support — skip rows already in Qdrant
    already_done = client.count(COLLECTION_NAME).count
    if already_done >= total:
        log.info("  all points already upserted — nothing to do")
        return
    if already_done > 0:
        log.info(f"  resuming from row {already_done:,}")
        df    = df.iloc[already_done:].reset_index(drop=True)
        texts = texts[already_done:]

    for chunk_start in range(0, len(texts), CHUNK_SIZE):
        chunk_texts = texts[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_df    = df.iloc[chunk_start : chunk_start + CHUNK_SIZE]
        global_start = already_done + chunk_start

        log.info(f"  embedding rows {global_start:,} – {global_start + len(chunk_texts):,} / {total:,}")

        embeddings = model.encode(
            chunk_texts,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=device,
        )
        torch.cuda.empty_cache()

        # Upsert in sub-batches
        for i in range(0, len(chunk_df), UPSERT_BATCH):
            batch_df  = chunk_df.iloc[i : i + UPSERT_BATCH]
            batch_emb = embeddings[i : i + UPSERT_BATCH]

            points = [
                PointStruct(
                    id=int(global_start + i + j),
                    vector=batch_emb[j].tolist(),
                    payload={
                        "parent_asin":   str(row.get("parent_asin") or ""),
                        "title":         str(row.get("title") or ""),
                        "price_numeric": _safe_float(row.get("price_numeric")),
                        "average_rating":_safe_float(row.get("average_rating")),
                        "rating_number": int(row.get("rating_number") or 0),
                        "main_category": str(row.get("main_category") or ""),
                        "sub_category":  _sub_category(row.get("categories")),
                        "brand":         _brand(row.get("details")),
                        "combined_text": str(row.get("combined_text") or ""),
                    },
                )
                for j, (_, row) in enumerate(batch_df.iterrows())
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)

        log.info(f"  upserted up to row {global_start + len(chunk_texts):,}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # GPU check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info(f"  GPU : {props.name}")
        log.info(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        log.warning("  No GPU detected — CPU embedding will be very slow")

    log.info("=== Step 1: Loading CSV ===")
    df = pd.read_csv(CSV_PATH)
    log.info(f"  {len(df):,} rows loaded")

    log.info("=== Step 2: Loading embedding model ===")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    log.info(f"  {EMBEDDING_MODEL} ready")

    log.info("=== Step 3: Connecting to Qdrant ===")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)
    setup_collection(client, recreate=False)

    log.info("=== Step 4: Embedding + upserting ===")
    embed_and_upsert(df, model, client, device)

    log.info("=== Step 5: Creating payload indexes ===")
    create_payload_indexes(client)

    final_count = client.count(COLLECTION_NAME).count
    print("\n" + "=" * 60)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Vectors    : {final_count:,}")
    print(f"Dim        : {VECTOR_DIM}  |  Distance: Cosine")
    print(f"Indexes    : price_numeric, average_rating, main_category, sub_category, brand")
    print("=" * 60)


if __name__ == "__main__":
    main()
