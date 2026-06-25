"""
Phase 1 — Data Preparation
Loads meta_Electronics.jsonl (1.6M items), cleans, stratified-samples to 200K,
builds combined_text for embedding, and saves products_200k.csv.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
JSONL_PATH   = Path(r"E:\Placement\Rag_Project\meta_Electronic.jsonl")
OUTPUT_CSV   = Path(r"E:\Placement\Rag_Project\products_200k.csv")
TARGET       = 200_000
MIN_PER_CAT  = 500   # floor: small categories get at least this many (if available)
RANDOM_STATE = 42

KEEP_FIELDS = [
    "parent_asin", "title", "description", "features",
    "price", "average_rating", "rating_number", "main_category", "store",
    "categories", "details",
]

MIN_RATING       = 3.5
MIN_RATING_COUNT = 10   # ignore products with too few ratings
MIN_DESC_LEN     = 30   # chars after joining description / features
MIN_CAT_SIZE     = 50   # drop categories with fewer products than this after cleaning

EXCLUDE_CATEGORIES = {
    "Books",
    "Movies & TV",
    "Grocery",
    "Buy a Kindle",
    "Gift Cards",
    "Magazine Subscriptions",
    "Handmade",
    "Collectible Coins",
    "Unique Finds",
}


# ── Step 1 — Load ─────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i % 200_000 == 0:
                log.info(f"  read {i:,} lines …")
            try:
                obj = json.loads(line)
                records.append({k: obj.get(k) for k in KEEP_FIELDS})
            except json.JSONDecodeError:
                continue
    log.info(f"  finished — {len(records):,} records parsed")
    return pd.DataFrame(records)


# ── Step 2 — Clean ────────────────────────────────────────────────────────────
def _join_list(val) -> str:
    if isinstance(val, list):
        return " ".join(str(v) for v in val if v)
    return str(val or "")


def _parse_price(val) -> float | None:
    if val is None:
        return None
    s = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(s) if s else None
    except ValueError:
        return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"  before cleaning: {len(df):,}")

    # title must exist
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]

    # rating >= 3.5, not null, and minimum review count
    df = df[df["average_rating"].notna() & (df["average_rating"] >= MIN_RATING)]
    df = df[df["rating_number"].notna() & (df["rating_number"] >= MIN_RATING_COUNT)]

    # must have main_category
    df = df[df["main_category"].notna() & (df["main_category"].str.strip() != "")]

    # description OR features must carry enough text
    df = df.copy()
    df["_desc_text"]     = df["description"].apply(_join_list)
    df["_feature_text"]  = df["features"].apply(_join_list)
    has_content = (
        (df["_desc_text"].str.len() >= MIN_DESC_LEN) |
        (df["_feature_text"].str.len() >= MIN_DESC_LEN)
    )
    df = df[has_content].drop(columns=["_desc_text", "_feature_text"])

    # parse price to float (keep nulls — some products have no price)
    df["price_numeric"] = df["price"].apply(_parse_price)

    log.info(f"  after cleaning:  {len(df):,}")
    return df.reset_index(drop=True)


# ── Step 3 — Stratified sample ────────────────────────────────────────────────
def stratified_sample(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    # drop non-electronics and noise categories
    df = df[~df[cat_col].isin(EXCLUDE_CATEGORIES)].copy()
    counts = df[cat_col].value_counts()
    valid_cats = counts[counts >= MIN_CAT_SIZE].index
    dropped = counts[counts < MIN_CAT_SIZE]
    if not dropped.empty:
        log.info(f"  dropping undersized categories: {dropped.to_dict()}")
    df = df[df[cat_col].isin(valid_cats)].reset_index(drop=True)

    counts  = df[cat_col].value_counts()
    n_total = len(df)
    pieces  = []

    log.info(f"  categories after filtering: {len(counts)}")
    log.info("\n" + counts.to_string())

    for cat, count in counts.items():
        proportional = int((count / n_total) * TARGET)
        n = max(proportional, min(MIN_PER_CAT, count))  # floor, but never exceed what's available
        n = min(n, count)
        pieces.append(
            df[df[cat_col] == cat].sample(n=n, random_state=RANDOM_STATE)
        )

    result = pd.concat(pieces).reset_index(drop=True)

    # If floor pushed us above TARGET, random-downsample uniformly
    if len(result) > TARGET:
        log.info(f"  total after floors = {len(result):,}; downsampling to {TARGET:,}")
        result = result.sample(n=TARGET, random_state=RANDOM_STATE).reset_index(drop=True)

    log.info(f"  final sample size: {len(result):,}")
    return result


# ── Step 4 — Build combined_text ──────────────────────────────────────────────
def build_combined_text(row) -> str:
    title    = str(row["title"] or "")
    category = str(row["main_category"] or "")
    desc     = _join_list(row["description"])[:600]
    features = _join_list(row["features"])[:400]
    price    = str(row["price_numeric"] or "")
    rating   = str(row["average_rating"] or "")

    cats = row.get("categories")
    sub_category = cats[-1] if isinstance(cats, list) and cats else ""

    details = row.get("details") or {}
    brand = str(details.get("Manufacturer") or details.get("Brand") or "")

    parts = []
    if title:        parts.append(f"Title: {title}")
    if brand:        parts.append(f"Brand: {brand}")
    if category:     parts.append(f"Category: {category}")
    if sub_category: parts.append(f"Sub-category: {sub_category}")
    if desc:         parts.append(f"Description: {desc}")
    if features:     parts.append(f"Features: {features}")
    if price:        parts.append(f"Price: {price}")
    if rating:       parts.append(f"Rating: {rating}")

    return ". ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} not found.\n"
            "If still extracting from .gz, wait for it to finish then re-run."
        )

    log.info("=== Step 1: Loading JSONL ===")
    df = load_jsonl(JSONL_PATH)

    log.info("=== Step 2: Cleaning ===")
    df_clean = clean(df)
    del df  # free ~2-4 GB RAM

    log.info("=== Step 3: Stratified sampling (200K, class-balanced) ===")
    df_sampled = stratified_sample(df_clean, "main_category")
    del df_clean

    log.info("=== Step 4: Building combined_text ===")
    df_sampled["combined_text"] = df_sampled.apply(build_combined_text, axis=1)

    log.info(f"=== Step 5: Saving → {OUTPUT_CSV} ===")
    df_sampled.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Saved {len(df_sampled):,} rows to {OUTPUT_CSV.name}")
    print("\nCategory distribution in final sample:")
    print(df_sampled["main_category"].value_counts().to_string())
    print("\nRating stats:")
    print(df_sampled["average_rating"].describe().round(2).to_string())
    print("\nPrice (numeric) coverage:")
    has_price = df_sampled["price_numeric"].notna().sum()
    print(f"  {has_price:,} / {len(df_sampled):,} records have a parseable price ({has_price/len(df_sampled):.1%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
