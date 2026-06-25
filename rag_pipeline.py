"""
Phase 3 — Core RAG Pipeline
Query → BGE embed → Qdrant retrieve → gpt-oss-120b generate → Response

Setup:
    pip install openai
    set NVIDIA_API_KEY=your_key_from_build.nvidia.com
"""

import logging
import os
import random
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Direction, FieldCondition, Filter, MatchValue, OrderBy, Range
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
NIM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
LLM_MODEL       = "openai/gpt-oss-120b"     # production generator (NVIDIA NIM)
LLM_MODEL_SMALL = "openai/gpt-oss-20b"      # smaller model for --compare mode
TOP_K           = 8

# BGE-small requires this prefix on queries (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = (
    "You are a helpful electronics shopping assistant. "
    "You are given up to 8 products retrieved from a catalog. "
    "Select the TOP 3 most suitable products for the user's query. "
    "For each of the 3 products: mention the product name, price, rating, and explain in 1-2 sentences why it fits the user's needs. "
    "Rank them from best to third-best match. "
    "If fewer than 3 products are genuinely relevant, only recommend those. "
    "If no product is a good match, say so honestly instead of forcing a recommendation. "
    "Keep your response structured and concise."
)


# ── Models (loaded once, reused across queries) ───────────────────────────────
def load_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  loading embedder on {device}")
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def load_llm() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY not set.\n"
            "Get a free key from https://build.nvidia.com → API Keys\n"
            "Then run:  set NVIDIA_API_KEY=your_key"
        )
    client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key, timeout=60.0)
    log.info(f"  LLM ready: {LLM_MODEL}")
    return client


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    embedder: SentenceTransformer,
    client: QdrantClient,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    category: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[dict]:

    if category is None:
        category = extract_category_from_query(query)

    query_vector = embedder.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    conditions = []
    if max_price is not None:
        conditions.append(
            FieldCondition(key="price_numeric", range=Range(gte=0.01, lte=max_price))
        )
    if min_rating is not None:
        conditions.append(
            FieldCondition(key="average_rating", range=Range(gte=min_rating))
        )
    if category is not None:
        log.info(f"  Category filter: {category}")
        conditions.append(
            FieldCondition(key="main_category", match=MatchValue(value=category))
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=conditions) if conditions else None,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "title":        h.payload.get("title", ""),
            "brand":        h.payload.get("brand", ""),
            "price":        h.payload.get("price_numeric"),
            "rating":       h.payload.get("average_rating"),
            "rating_count": h.payload.get("rating_number", 0),
            "category":     h.payload.get("main_category", ""),
            "sub_category": h.payload.get("sub_category", ""),
            "score":        round(h.score, 4),
        }
        for h in results.points
    ]


# ── Context formatting ────────────────────────────────────────────────────────
def format_context(products: list[dict]) -> str:
    if not products:
        return "No products found matching your criteria."

    lines = []
    for i, p in enumerate(products, 1):
        price_str  = f"${p['price']:.2f}" if p["price"] else "Price not listed"
        cat_str    = p["sub_category"] or p["category"]
        brand_str  = p["brand"] or "N/A"
        lines.append(
            f"{i}. {p['title']}\n"
            f"   Brand: {brand_str}  |  Price: {price_str}  |  "
            f"Rating: {p['rating']}/5 ({p['rating_count']} reviews)  |  "
            f"Category: {cat_str}"
        )

    return "\n\n".join(lines)


# ── Generation ────────────────────────────────────────────────────────────────
def generate(query: str, context: str, llm: OpenAI, model_name: str = LLM_MODEL) -> str:
    prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Products:\n{context}\n\n"
        f"Provide your recommendation:"
    )
    stream = llm.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
        extra_body={"reasoning_effort": "low"},
        stream=True,   # keeps Ctrl+C responsive on Windows
    )
    chunks = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    return "".join(chunks)


# ── Model comparison ─────────────────────────────────────────────────────────
def compare_models(query: str, context: str, llm: OpenAI) -> dict:
    import time
    results = {}
    for model_name, label in [
        (LLM_MODEL,       "gpt-oss-120b"),
        (LLM_MODEL_SMALL, "gpt-oss-20b"),
    ]:
        print(f"  Calling {label}...")
        start    = time.time()
        response = generate(query, context, llm, model_name)
        elapsed  = round(time.time() - start, 2)
        results[label] = {"response": response, "latency_sec": elapsed}
    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────
def recommend(
    query: str,
    embedder: SentenceTransformer,
    client: QdrantClient,
    llm: OpenAI,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    category: Optional[str] = None,
) -> dict:

    products = retrieve(query, embedder, client, max_price, min_rating, category)
    context  = format_context(products)
    response = generate(query, context, llm)

    return {
        "query":    query,
        "filters":  {"max_price": max_price, "min_rating": min_rating, "category": category},
        "products": products,
        "response": response,
    }


# ── Query parsing ────────────────────────────────────────────────────────────
def extract_price_from_query(query: str) -> Optional[float]:
    patterns = [
        r'under\s+\$?(\d+(?:\.\d+)?)',
        r'less\s+than\s+\$?(\d+(?:\.\d+)?)',
        r'below\s+\$?(\d+(?:\.\d+)?)',
        r'max(?:imum)?\s+\$?(\d+(?:\.\d+)?)',
        r'budget\s+of\s+\$?(\d+(?:\.\d+)?)',
        r'\$(\d+(?:\.\d+)?)\s+or\s+less',
        r'within\s+\$?(\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return float(match.group(1))
    return None


def extract_rating_from_query(query: str) -> Optional[float]:
    """Pull a minimum-rating constraint out of natural language.

    Handles explicit numbers ("rating above 4.5", "4+ stars") and qualitative
    phrasing ("better ratings", "highly rated") which maps to a 4.5 threshold.
    """
    q = query.lower()

    numeric_patterns = [
        r'(?:rating|rated|stars?)\s*(?:of\s*)?(?:above|over|at\s+least|greater\s+than|more\s+than|>=?)\s*(\d(?:\.\d)?)',
        r'(?:above|over|at\s+least|greater\s+than|more\s+than)\s*(\d(?:\.\d)?)\s*(?:star|rating)',
        r'(\d(?:\.\d)?)\s*\+\s*(?:star|rating)',
        r'(\d(?:\.\d)?)\s*stars?\s*(?:or\s+(?:higher|above|more|better))',
    ]
    for pattern in numeric_patterns:
        match = re.search(pattern, q)
        if match:
            return min(float(match.group(1)), 5.0)

    qualitative = (
        "better rating", "better ratings", "higher rating", "higher ratings",
        "high rating", "high ratings", "high rated", "high-rated",
        "highly rated", "well rated", "well-rated", "top rated", "top-rated",
        "best rated", "good rating", "good ratings", "great rating", "great ratings",
    )
    if any(phrase in q for phrase in qualitative):
        return 4.5

    return None


# ── Category-aware filtering ─────────────────────────────────────────────────
# Maps query keywords → exact Qdrant main_category values.
# Only includes mappings verified against actual catalog data — categories where
# the product type reliably lands in ONE category. Ambiguous types (headphones,
# speakers, earbuds) are intentionally excluded to avoid over-filtering.
_CATEGORY_MAP: dict[str, str] = {
    # multi-word phrases first (most specific)
    "action camera":   "Camera & Photo",     # confirmed
    "dash cam":        "Car Electronics",
    "home theater":    "Home Audio & Theater",
    "gaming laptop":   "Computers",          # confirmed
    "gaming keyboard": "Computers",          # confirmed
    "gaming mouse":    "All Electronics",    # mice live in All Electronics
    "usb hub":         "Computers",
    "hard drive":      "Computers",
    "fire tv":         "Amazon Fire TV",
    "fitness tracker": "All Electronics",
    # single words — only where catalog data confirms a dominant category
    "laptop":          "Computers",          # confirmed
    "notebook":        "Computers",
    "macbook":         "Computers",
    "chromebook":      "Computers",
    "keyboard":        "Computers",          # confirmed
    "monitor":         "Computers",          # confirmed
    "webcam":          "Computers",          # confirmed (NOT Camera & Photo)
    "mouse":           "All Electronics",    # confirmed (NOT Computers)
    "ssd":             "Computers",
    "dslr":            "Camera & Photo",     # confirmed
    "mirrorless":      "Camera & Photo",     # confirmed
    "smartwatch":      "All Electronics",    # confirmed
    "wearable":        "All Electronics",
    "phone":           "Cell Phones & Accessories",
    "smartphone":      "Cell Phones & Accessories",
    "gps":             "GPS & Navigation",
    "kindle":          "Amazon Devices",
    "guitar":          "Musical Instruments",
    "piano":           "Musical Instruments",
    # Intentionally NOT mapped (spread across 3+ categories in real data):
    #   headphone, headset, earbud, earphone → All Electronics + Cell Phones + Home Audio
    #   speaker, soundbar                    → Home Audio + All Electronics + Industrial
    #   camera (generic)                     → Amazon Devices + Camera & Photo + Amazon Home
}


def extract_category_from_query(query: str) -> Optional[str]:
    """Return a Qdrant main_category when the query mentions a known product type.

    Checks multi-word phrases before single words so 'gaming laptop' maps to
    Computers rather than accidentally matching a shorter keyword.
    """
    q = query.lower()
    for phrase in sorted(_CATEGORY_MAP, key=len, reverse=True):
        if phrase in q:
            return _CATEGORY_MAP[phrase]
    return None


# ── Capability query detection ───────────────────────────────────────────────
_CAPABILITY_TRIGGERS = (
    "what can you", "what do you", "what products", "what categories",
    "what do you sell", "what do you have", "what can i ask",
    "what can i search", "what are you", "what kind of", "what type of",
    "tell me about yourself", "what items", "what goods", "what range",
    "what brands", "how can you help", "how do you work",
    "what do you recommend", "what's available", "whats available",
    "what is available", "what electronics", "what can i buy",
)


def is_capability_query(text: str) -> bool:
    t = text.lower().strip()
    return any(phrase in t for phrase in _CAPABILITY_TRIGGERS)


_EXAMPLE_QUERIES = [
    "*'Recommend Sony wireless headphones under $150 with at least 4.5 stars'*",
    "*'Best gaming laptops under $1000'*",
    "*'Logitech mechanical keyboards under $80 with good reviews'*",
    "*'4K monitors under $400 highly rated'*",
    "*'Noise cancelling headphones under $200'*",
    "*'Budget webcam for video calls under $50'*",
    "*'Best rated SSDs under $100'*",
    "*'Wireless gaming mouse under $60 with 4+ stars'*",
    "*'Portable bluetooth speakers under $80'*",
    "*'USB-C hubs under $40 well rated'*",
    "*'Gaming chairs under $300 with at least 4 stars'*",
    "*'Best mirrorless cameras under $800'*",
]


def get_catalog_summary(qdrant: QdrantClient, collection: str = COLLECTION_NAME) -> str:
    """Query Qdrant for live catalog stats and return a formatted answer."""
    try:
        total = qdrant.count(collection).count

        cat_result   = qdrant.facet(collection_name=collection, key="main_category", limit=25)
        categories   = [h.value for h in cat_result.hits if h.value]

        brand_result = qdrant.facet(collection_name=collection, key="brand", limit=2000)
        brand_count  = len([h for h in brand_result.hits if h.value])

        cheap_pts, _ = qdrant.scroll(
            collection_name=collection, limit=1,
            order_by=OrderBy(key="price_numeric", direction=Direction.ASC),
            scroll_filter=Filter(must=[FieldCondition(key="price_numeric", range=Range(gte=1.0))]),
            with_payload=["price_numeric"], with_vectors=False,
        )
        exp_pts, _ = qdrant.scroll(
            collection_name=collection, limit=1,
            order_by=OrderBy(key="price_numeric", direction=Direction.DESC),
            with_payload=["price_numeric"], with_vectors=False,
        )
        min_price = cheap_pts[0].payload.get("price_numeric", 0) if cheap_pts else 0
        max_price = exp_pts[0].payload.get("price_numeric", 0)   if exp_pts   else 0

        cat_str = ", ".join(categories[:12])
        if len(categories) > 12:
            cat_str += f" … and {len(categories) - 12} more"

        return (
            f"I'm your personal electronics shopping assistant backed by a catalog of "
            f"**{total:,} products** across **{len(categories)} categories** "
            f"from **{brand_count}+ brands**, priced from **${min_price:.0f}** to **${max_price:.0f}**.\n\n"
            f"**Categories I cover:** {cat_str}\n\n"
            "You can ask me for recommendations using any combination of:\n"
            "- **Category** — e.g. *'wireless headphones'*, *'gaming laptop'*\n"
            "- **Budget** — e.g. *'under $100'*, *'between $50 and $200'*\n"
            "- **Rating** — e.g. *'4+ stars'*, *'highly rated'*\n"
            "- **Brand** — e.g. *'Sony speakers'*, *'Logitech keyboard'*\n\n"
            f"Example: {random.choice(_EXAMPLE_QUERIES)}"
        )
    except Exception as e:
        log.warning(f"  catalog summary failed: {e}")
        return (
            "I can recommend electronics products from a large catalog covering laptops, "
            "headphones, keyboards, monitors, and more. "
            f"Try: {random.choice(_EXAMPLE_QUERIES)}"
        )


# ── Interactive CLI ───────────────────────────────────────────────────────────
def _parse_optional_float(prompt_text: str) -> Optional[float]:
    val = input(prompt_text).strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        print("  Invalid number — skipping filter.")
        return None


def main():
    import sys
    import time

    compare_mode = "--compare" in sys.argv

    log.info("=== Initialising RAG pipeline ===")
    embedder = load_embedder()
    client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    llm      = load_llm()

    count = client.count(COLLECTION_NAME).count
    log.info(f"  Qdrant ready — {count:,} products indexed")

    if compare_mode:
        print("\n" + "=" * 60)
        print("  [DEV MODE] Model Comparison — gpt-oss-120b vs gpt-oss-20b")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  E-Commerce RAG Assistant")
        print("  Type 'quit' to exit")
        print("=" * 60)

    while True:
        print()
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if is_capability_query(query):
            print()
            print(get_catalog_summary(client))
            continue

        auto_price = extract_price_from_query(query)
        if auto_price:
            override = input(f"Detected price limit ${auto_price:.0f} — press Enter to confirm or type a new amount: ").strip()
            max_price = float(override) if override else auto_price
        else:
            max_price = _parse_optional_float("Max price in $ (Enter to skip): ")

        min_rating = _parse_optional_float("Min rating e.g. 4.0 (Enter to skip): ")

        print("\nSearching...\n")
        products = retrieve(query, embedder, client, max_price, min_rating)
        context  = format_context(products)

        if compare_mode:
            comparison = compare_models(query, context, llm)
            print()
            for label, data in comparison.items():
                print(f"{'─' * 60}")
                print(f"  {label}  |  latency: {data['latency_sec']}s")
                print(f"{'─' * 60}")
                print(data["response"])
        else:
            start    = time.time()
            response = generate(query, context, llm)
            elapsed  = round(time.time() - start, 2)
            print(f"Assistant:\n{response}")
            print(f"\n[{elapsed}s | {len(products)} products retrieved | "
                  f"Top score: {products[0]['score'] if products else 'N/A'}]")

        print("=" * 60)


if __name__ == "__main__":
    main()
