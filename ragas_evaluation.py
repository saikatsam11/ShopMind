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
NIM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
GENERATOR_MODEL = "openai/gpt-oss-120b"
EVALUATOR_MODEL = "meta/llama-3.3-70b-instruct"
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
    "Keep your response structured and concise."
)

# ── 23 test queries (20 standard + 3 edge cases) ───────────────────────────────
RAGAS_QUERIES = [
    {"question": "Recommend wireless headphones for gym workout",
     "ground_truth": "Good gym headphones should be sweat-resistant, have a secure fit, Bluetooth connectivity, and at least 6 hours battery life.",
     "filters": {}},
    {"question": "Best laptop for video editing under $1500",
     "ground_truth": "A video editing laptop needs a powerful CPU, dedicated GPU with at least 4GB VRAM, 16GB+ RAM, fast SSD, and a color-accurate display.",
     "filters": {"max_price": 1500.0}},
    {"question": "Affordable mechanical keyboard for programming",
     "ground_truth": "A programming keyboard should have tactile or linear switches, N-key rollover, durable build, and ideally be tenkeyless. Price range $50-$150.",
     "filters": {}},
    {"question": "Noise cancelling headphones for travel",
     "ground_truth": "Travel headphones need active noise cancellation, comfortable over-ear design, 20+ hours battery, and foldable form factor.",
     "filters": {}},
    {"question": "Budget webcam for video conferencing under $80",
     "ground_truth": "A conferencing webcam should offer 1080p resolution, built-in microphone, USB plug-and-play, and decent low-light performance.",
     "filters": {"max_price": 80.0}},
    {"question": "Gaming mouse under $60 with good reviews",
     "ground_truth": "A gaming mouse needs a high DPI sensor, programmable buttons, ergonomic design, and durable switches. 4+ star rating is a good indicator.",
     "filters": {"max_price": 60.0}},
    {"question": "Portable Bluetooth speaker with good bass",
     "ground_truth": "A portable speaker should have strong bass, IPX4 water resistance, 12+ hours battery, and Bluetooth 5.0 connectivity.",
     "filters": {}},
    {"question": "4K monitor for graphic design",
     "ground_truth": "A graphic design monitor needs 4K resolution, 99%+ sRGB coverage, factory calibration, IPS panel, and at least 27 inches.",
     "filters": {}},
    {"question": "Smartwatch for fitness tracking",
     "ground_truth": "A fitness smartwatch should track heart rate, steps, calories, and sleep, with GPS, water resistance, and at least 5-day battery.",
     "filters": {}},
    {"question": "Budget DSLR camera for beginner photographers",
     "ground_truth": "A beginner DSLR needs an APS-C sensor, interchangeable lenses, easy-to-use interface, and should come with a kit lens under $800.",
     "filters": {}},
    {"question": "Wireless gaming headset for PC",
     "ground_truth": "A PC gaming headset needs surround sound, noise-cancelling mic, comfortable over-ear design, low-latency wireless, and 15+ hours battery.",
     "filters": {}},
    {"question": "External SSD 1TB for fast data transfer",
     "ground_truth": "A 1TB external SSD needs USB 3.1 or USB-C for fast speeds (400MB/s+), compact form factor, and cross-platform compatibility.",
     "filters": {}},
    {"question": "USB hub for MacBook with multiple ports",
     "ground_truth": "A MacBook USB hub needs USB-C connection, multiple USB-A ports, HDMI output, SD card reader, and power delivery passthrough.",
     "filters": {}},
    {"question": "Gaming laptop under $800",
     "ground_truth": "A budget gaming laptop needs at least a GTX 1650 GPU, Intel i5 or Ryzen 5 CPU, 8GB RAM, 512GB SSD, and 1080p 60Hz+ display.",
     "filters": {"max_price": 800.0}},
    {"question": "Action camera for outdoor adventures",
     "ground_truth": "An action camera should be waterproof, shoot 4K at 60fps, have image stabilization, wide-angle lens, and be compact and mountable.",
     "filters": {}},
    {"question": "Wireless earbuds with long battery life",
     "ground_truth": "Good wireless earbuds should offer 6+ hours per charge, 24+ total with case, ANC or good isolation, and IPX4 water resistance.",
     "filters": {}},
    {"question": "Mechanical keyboard under $50 for office use",
     "ground_truth": "An affordable office keyboard should have quiet tactile switches, full-size or TKL layout, and comfortable keycaps.",
     "filters": {"max_price": 50.0}},
    {"question": "Highly rated Sony wireless headphones",
     "ground_truth": "Sony wireless headphones are known for excellent sound quality, reliable ANC, comfortable design, long battery, and LDAC audio support.",
     "filters": {}},
    {"question": "Best rated wireless mouse for productivity",
     "ground_truth": "A productivity wireless mouse should be ergonomic, have a precise optical sensor, 3+ month battery, and multi-device connectivity.",
     "filters": {}},
    {"question": "Affordable monitor for coding under $300",
     "ground_truth": "A coding monitor needs 1080p+, 24-27 inch IPS panel, adjustable stand, and eye care features like flicker-free and blue light filter.",
     "filters": {"max_price": 300.0}},
    # ── Edge case 1: out-of-catalog query ──────────────────────────────────────
    {"question": "Recommend waterproof hiking boots for trail running",
     "ground_truth": "The assistant should recognize hiking boots are not in the electronics catalog and honestly decline rather than recommending irrelevant products.",
     "filters": {}},
    # ── Edge case 2: vague query ────────────────────────────────────────────────
    {"question": "I want something good for music",
     "ground_truth": "A vague music query could match wireless headphones, Bluetooth speakers, or earbuds. The assistant should recommend the most popular category with good ratings.",
     "filters": {}},
    # ── Edge case 3: multi-constraint query ────────────────────────────────────
    {"question": "Sony wireless headphones under $100 with at least 4.5 stars",
     "ground_truth": "Sony wireless headphones under $100 with 4.5+ stars should offer good sound quality, comfortable fit, and reliable Bluetooth. All three constraints must be respected.",
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

    print("  Loading embedder...")
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("  Connecting to Qdrant...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    print("  Loading generator LLM (gpt-oss-120b)...")
    llm = OpenAI(base_url=NIM_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"], timeout=60.0)

    samples = []
    print(f"\n  Generating answers for {len(RAGAS_QUERIES)} queries...")

    for i, item in enumerate(RAGAS_QUERIES, 1):
        q = item["question"]
        f = item["filters"]
        print(f"    [{i:02d}/{len(RAGAS_QUERIES)}] {q[:65]}...", end=" ", flush=True)

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
            price = f"${p.get('price_numeric') or 0:.2f}"
            brand = p.get("brand") or "N/A"
            cat   = p.get("sub_category") or p.get("main_category", "")
            ctx_list.append(
                f"Product: {p.get('title', '')}\n"
                f"Brand: {brand} | Price: {price} | "
                f"Rating: {p.get('average_rating')}/5 ({p.get('rating_number', 0)} reviews) | "
                f"Category: {cat}"
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
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    n_calls = len(samples) * (TOP_K + 4)
    print(f"  Running RAGAS on {len(samples)} queries (~{n_calls} evaluator LLM calls)")
    print("  max_workers=2 to stay within NIM rate limits — expect 15-20 min\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=2, timeout=300),
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
        print("Phase 1 — Generating RAG answers (this calls gpt-oss-120b 23 times)...")
        samples = generate_dataset()
        with open(DATASET_PATH, "w", encoding="utf-8") as fh:
            json.dump(samples, fh, indent=2, ensure_ascii=False)
        print(f"\n  Dataset saved → {DATASET_PATH.name}")

    print("\nPhase 2 — Running RAGAS evaluation (nemotron-3-ultra evaluator)...")
    run_ragas(samples)


if __name__ == "__main__":
    main()
