"""
Phase 7 — RAGAS Evaluation  (ragas >= 0.4.x API)
Two-phase design:
  Phase 1 — generate_dataset()  : run 23 queries through the RAG pipeline,
                                   save answers + contexts to ragas_dataset.json
  Phase 2 — run_ragas()         : load dataset, evaluate with nemotron-3-ultra

Re-running only redoes Phase 1 if ragas_dataset.json is missing.

Generator : openai/gpt-oss-120b               (NVIDIA NIM)
Evaluator : nvidia/nemotron-3-ultra-550b-a55b (NVIDIA NIM, different model → no self-grading bias)

Required packages (install in this order):
    pip install ragas langchain-openai langchain-huggingface --upgrade
    pip install sentence-transformers --upgrade --force-reinstall
    pip install numpy --upgrade
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION_NAME = "products"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_BASE_URL   = "https://api.groq.com/openai/v1"
NIM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
# Generator on Groq, judge on NIM: the judge issues ~7x more calls, and NIM
# has no daily token cap (40 RPM only) while Groq free tier caps at 200K TPD.
# Different labs, so the generator never grades its own output.
GENERATOR_MODEL = "qwen/qwen3.8-27b"      # Groq
EVALUATOR_MODEL = "openai/gpt-oss-20b"    # NVIDIA NIM
QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "
TOP_K           = 3
DATASET_PATH    = Path(__file__).parent / "ragas_dataset.json"

# ── Category-aware filtering (mirrors 03_rag_pipeline._CATEGORY_MAP) ──────────
_CATEGORY_MAP: dict[str, str] = {
    "action camera":   "Camera & Photo",
    "dash cam":        "Car Electronics",
    "home theater":    "Home Audio & Theater",
    "gaming laptop":   "Computers",
    "gaming keyboard": "Computers",
    "gaming mouse":    "All Electronics",
    "usb hub":         "Computers",
    "hard drive":      "Computers",
    "fire tv":         "Amazon Fire TV",
    "fitness tracker": "All Electronics",
    "laptop":          "Computers",
    "notebook":        "Computers",
    "macbook":         "Computers",
    "chromebook":      "Computers",
    "keyboard":        "Computers",
    "monitor":         "Computers",
    "webcam":          "Computers",
    "mouse":           "All Electronics",
    "ssd":             "Computers",
    "dslr":            "Camera & Photo",
    "mirrorless":      "Camera & Photo",
    "smartwatch":      "All Electronics",
    "wearable":        "All Electronics",
    "phone":           "Cell Phones & Accessories",
    "smartphone":      "Cell Phones & Accessories",
    "gps":             "GPS & Navigation",
    "kindle":          "Amazon Devices",
    "guitar":          "Musical Instruments",
    "piano":           "Musical Instruments",
}


def _extract_category(query: str) -> Optional[str]:
    q = query.lower()
    for phrase in sorted(_CATEGORY_MAP, key=len, reverse=True):
        if phrase in q:
            return _CATEGORY_MAP[phrase]
    return None


SYSTEM_PROMPT = (
    "You are a helpful electronics shopping assistant. "
    "You are given up to 3 products retrieved from a catalog. "
    "Recommend the most suitable ones for the user's query. "
    "For each product: mention name, price, rating, and 1-2 sentences on why it fits. "
    "If no product is a good match, say so honestly. "
    "Ground every claim in the product information provided. "
    "Do not add specifications, opinions, marketing language, or comparisons "
    "that are not explicitly stated in the product text. "
    "If a detail the user asked about is absent from the product text, say it is "
    "not specified rather than inferring it. "
    "If a price is not listed, say so - never state $0.00 as a price. "
    "Keep your response structured and concise."
)

# ── 23 test queries (20 standard + 3 edge cases) ───────────────────────────────
RAGAS_QUERIES = [
    {"question": "Recommend wireless headphones for gym workout",
     "ground_truth": "The recommended products are wireless Bluetooth headphones or earbuds whose listings describe them as sweatproof or water-resistant and intended for sports, running, or gym use, with a secure or ear-hook fit and a good user rating.",
     "filters": {}},
    {"question": "Best laptop for video editing under $1500",
     "ground_truth": "The recommended products are laptops priced under $1500 whose listings indicate a capable processor, sufficient RAM, and SSD storage suitable for video editing work.",
     "filters": {"max_price": 1500.0}},
    {"question": "Affordable mechanical keyboard for programming",
     "ground_truth": "The recommended products are mechanical keyboards priced roughly $50 to $150 whose listings mention mechanical switches and a layout suited to typing or programming, such as TKL, 75 percent, or full-size.",
     "filters": {}},
    {"question": "Noise cancelling headphones for travel",
     "ground_truth": "The recommended products are headphones whose listings describe active noise cancellation and an over-ear or foldable design suited to travel, with long battery life and a good rating.",
     "filters": {}},
    {"question": "Budget webcam for video conferencing under $80",
     "ground_truth": "The recommended products are webcams priced under $80 whose listings mention 1080p or HD video, a built-in microphone, and USB plug-and-play connectivity.",
     "filters": {"max_price": 80.0}},
    {"question": "Gaming mouse under $60 with good reviews",
     "ground_truth": "The recommended products are gaming mice priced under $60 with a rating of 4 stars or higher, whose listings mention an adjustable DPI sensor, programmable buttons, or ergonomic gaming design.",
     "filters": {"max_price": 60.0}},
    {"question": "Portable Bluetooth speaker with good bass",
     "ground_truth": "The recommended products are portable Bluetooth speakers whose listings mention bass performance, water resistance, and battery life, with a good user rating.",
     "filters": {}},
    {"question": "4K monitor for graphic design",
     "ground_truth": "The recommended products are monitors whose listings state 4K or UHD resolution and an IPS panel, at a screen size suitable for design work, with colour reproduction mentioned in the description.",
     "filters": {}},
    {"question": "Smartwatch for fitness tracking",
     "ground_truth": "The recommended products are smartwatches or fitness trackers whose listings mention heart-rate monitoring, activity or sleep tracking, and water resistance, with a good user rating.",
     "filters": {}},
    {"question": "Budget DSLR camera for beginner photographers",
     "ground_truth": "The recommended products are DSLR or mirrorless cameras priced under $800 whose listings mention interchangeable lenses or an included kit lens and beginner-friendly operation.",
     "filters": {}},
    {"question": "Wireless gaming headset for PC",
     "ground_truth": "The recommended products are wireless gaming headsets compatible with PC whose listings mention surround sound, a noise-cancelling or detachable microphone, and an over-ear design.",
     "filters": {}},
    {"question": "External SSD 1TB for fast data transfer",
     "ground_truth": "The recommended products are 1TB external solid-state drives whose listings mention USB 3.1, USB-C, or high transfer speeds, in a compact portable form factor.",
     "filters": {}},
    {"question": "USB hub for MacBook with multiple ports",
     "ground_truth": "The recommended products are USB-C hubs or docks compatible with MacBook whose listings mention multiple USB-A ports and additional outputs such as HDMI, SD card reader, or power delivery.",
     "filters": {}},
    {"question": "Gaming laptop under $800",
     "ground_truth": "The recommended products are laptops priced under $800 whose listings mention a dedicated graphics card and gaming use, with at least 8GB of RAM and SSD storage.",
     "filters": {"max_price": 800.0}},
    {"question": "Action camera for outdoor adventures",
     "ground_truth": "The recommended products are action cameras whose listings mention waterproof or water-resistant construction, 4K video, image stabilisation, and mounting accessories for outdoor use.",
     "filters": {}},
    {"question": "Wireless earbuds with long battery life",
     "ground_truth": "The recommended products are true wireless earbuds whose listings state playtime per charge and additional charge from the case, with water resistance and a good user rating.",
     "filters": {}},
    {"question": "Mechanical keyboard under $50 for office use",
     "ground_truth": "The recommended products are mechanical keyboards priced under $50 whose listings mention quiet or tactile switches and a full-size or tenkeyless layout suitable for office typing.",
     "filters": {"max_price": 50.0}},
    {"question": "Highly rated Sony wireless headphones",
     "ground_truth": "The recommended products are Sony-branded wireless headphones with a high user rating, whose listings mention Bluetooth connectivity and sound-quality features such as noise cancellation.",
     "filters": {}},
    {"question": "Best rated wireless mouse for productivity",
     "ground_truth": "The recommended products are wireless mice with high user ratings whose listings mention ergonomic design, precise tracking, and long battery life or multi-device pairing.",
     "filters": {}},
    {"question": "Affordable monitor for coding under $300",
     "ground_truth": "The recommended products are monitors priced under $300 whose listings state 1080p or higher resolution, a 24 to 27 inch IPS panel, and eye-care features such as flicker-free or blue-light filtering.",
     "filters": {"max_price": 300.0}},
    # ── Edge case 1: out-of-catalog query ──────────────────────────────────────
    {"question": "Recommend waterproof hiking boots for trail running",
     "ground_truth": "The retrieved products are electronics and none match hiking boots or trail-running footwear, so the assistant states that no suitable product exists in the catalogue rather than recommending unrelated items.",
     "filters": {}},
    # ── Edge case 2: vague query ────────────────────────────────────────────────
    {"question": "I want something good for music",
     "ground_truth": "The recommended products are audio devices such as wireless headphones, earbuds, or Bluetooth speakers with strong user ratings, presented as options for listening to music.",
     "filters": {}},
    # ── Edge case 3: multi-constraint query ────────────────────────────────────
    {"question": "Sony wireless headphones under $100 with at least 4.5 stars",
     "ground_truth": "The recommended products are Sony-branded wireless headphones priced under $100 with a user rating of at least 4.5, whose listings mention Bluetooth connectivity and sound quality.",
     "filters": {"max_price": 100.0, "min_rating": 4.5}},
]


# ── Phase 1: build dataset ──────────────────────────────────────────────────────
def generate_dataset() -> list[dict]:
    """Run all queries through the RAG pipeline and return a list of samples."""
    import torch
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
    from rag_pipeline import extract_details

    print("  Loading embedder...")
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("  Connecting to Qdrant...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    print("  Loading generator LLM (gpt-oss-120b)...")
    llm = OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"], timeout=120.0)

    samples = []
    # RAGAS_LIMIT=3 runs a cheap smoke test instead of the full set.
    _limit  = int(os.environ.get("RAGAS_LIMIT", "0"))
    queries = RAGAS_QUERIES[:_limit] if _limit else RAGAS_QUERIES
    print(f"\n  Generating answers for {len(queries)} queries...")

    for i, item in enumerate(queries, 1):
        q = item["question"]
        f = item["filters"]
        print(f"    [{i:02d}/{len(queries)}] {q[:65]}...", end=" ", flush=True)

        # Retrieve
        vec        = embedder.encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()
        conditions = []
        if f.get("max_price"):
            conditions.append(FieldCondition(key="price_numeric", range=Range(gte=0.01, lte=f["max_price"])))
        if f.get("min_rating"):
            conditions.append(FieldCondition(key="average_rating", range=Range(gte=f["min_rating"])))
        category = _extract_category(q)
        if category:
            conditions.append(FieldCondition(key="main_category", match=MatchValue(value=category)))
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(must=conditions) if conditions else None,
            limit=TOP_K,
            with_payload=True,
        )

        # Format context — list of strings (one per product) for RAGAS
        ctx_list = []
        for h in results.points:
            p = h.payload
            _pn   = p.get("price_numeric")
            price = f"${_pn:.2f}" if _pn else "Price not listed"
            brand = p.get("brand") or "N/A"
            cat   = p.get("sub_category") or p.get("main_category", "")
            ctx_list.append(
                f"Product: {p.get('title', '')}\n"
                f"Brand: {brand} | Price: {price} | "
                f"Rating: {p.get('average_rating')}/5 ({p.get('rating_number', 0)} reviews) | "
                f"Category: {cat}"
                + (lambda d: f"\n{d}" if d else "")(
                    extract_details(p.get("combined_text", ""))
                )
            )

        # Generate answer
        ctx_joined = "\n\n".join(ctx_list) if ctx_list else "No products found."
        stream = llm.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"User Query: {q}\n\nRetrieved Products:\n{ctx_joined}\n\nProvide your recommendation:"},
            ],
            temperature=0.1, max_tokens=1024,
            extra_body={"reasoning_effort": "low"},
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        answer = "".join(chunks)

        samples.append({
            "question":     q,
            "answer":       answer,
            "contexts":     ctx_list if ctx_list else ["No products found."],
            "ground_truth": item["ground_truth"],
        })
        print("done")

    return samples


# ── Phase 2: RAGAS evaluation (ragas 0.4.x native API) ───────────────────────
def run_ragas(samples: list[dict]):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    # Build ragas 0.4 dataset — field names changed from 0.1.x
    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            retrieved_contexts=s["contexts"],
            response=s["answer"],
            reference=s["ground_truth"],
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    print(f"\n  Setting up evaluator: {EVALUATOR_MODEL}")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model=EVALUATOR_MODEL,
        base_url=NIM_BASE_URL,
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=0,
        max_tokens=4096,
        timeout=300,
    ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    # evaluate() sets llm/embeddings on MetricWithLLM/MetricWithEmbeddings
    # metrics and calls metric.init(run_config) internally — don't pre-init
    # NIM supports n>1, so strictness stays at the RAGAS default of 3.
    metrics = [Faithfulness(), AnswerRelevancy(strictness=3),
               ContextPrecision(), ContextRecall()]

    n_calls = len(samples) * (TOP_K + 4)
    print(f"  Running RAGAS on {len(samples)} queries (~{n_calls} evaluator LLM calls)")
    print("  generator: Groq | judge: NIM (no daily cap, 40 RPM) | max_workers=3" + chr(10))

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=3, timeout=300, max_retries=10, max_wait=60),
    )

    df = result.to_pandas()

    # Column names in ragas 0.4: faithfulness, answer_relevancy,
    # llm_context_precision, llm_context_recall
    metric_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]

    sep = "=" * 60
    print(f"\n{sep}")
    print("  RAGAS Evaluation Results")
    print(sep)

    display_cols = ["user_input"] + metric_cols
    print(df[display_cols].to_string(index=False, max_colwidth=45))
    print(sep)

    print("\n  Per-metric averages:")
    means = df[metric_cols].mean()
    for col in metric_cols:
        print(f"    {col:<30} : {means[col]:.3f}")
    overall = means.mean()
    print(f"    {'─' * 42}")
    print(f"    {'RAGAS Score (avg of all)':<30} : {overall:.3f}")
    print(sep)

    out = Path(__file__).parent / "ragas_results.csv"
    df.to_csv(out, index=False)
    print(f"\n  Results saved → {out.name}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print("=== Phase 7: RAGAS Evaluation ===\n")

    if DATASET_PATH.exists():
        print(f"Found existing dataset: {DATASET_PATH.name}")
        print("  (Delete it to regenerate answers from the RAG pipeline.)\n")
        with open(DATASET_PATH, encoding="utf-8") as fh:
            samples = json.load(fh)
    else:
        print("Phase 1 — Generating RAG answers (23 times)...")
        samples = generate_dataset()
        with open(DATASET_PATH, "w", encoding="utf-8") as fh:
            json.dump(samples, fh, indent=2, ensure_ascii=False)
        print(f"\n  Dataset saved → {DATASET_PATH.name}")

    print("\nPhase 2 — Running RAGAS evaluation (nemotron-3-ultra evaluator)...")
    run_ragas(samples)


if __name__ == "__main__":
    main()
