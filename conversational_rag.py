"""
Phase 4 — Conversational RAG
Adds on top of Phase 3:
  - Multi-turn chat history
  - Query rewriting  (resolves follow-ups like "cheaper ones", "show more")
  - MMR re-ranking   (fixes brand diversity issue)
  - Session preference memory (price/rating persist across turns)
  - Hybrid search    (dense + BM25 via --hybrid flag)

Run:
    python 04_conversational_rag.py             # dense only
    python 04_conversational_rag.py --hybrid    # dense + BM25
"""

import importlib.util
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, FusionQuery, Fusion,
    MatchValue, Prefetch, Range, SparseVector,
)

load_dotenv()

# ── Import shared utilities from rag_pipeline ─────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "rag_pipeline", Path(__file__).parent / "rag_pipeline.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_embedder            = _mod.load_embedder
load_llm                 = _mod.load_llm
format_context           = _mod.format_context
extract_price_from_query  = _mod.extract_price_from_query
extract_rating_from_query   = _mod.extract_rating_from_query
extract_category_from_query = _mod.extract_category_from_query
is_capability_query         = _mod.is_capability_query
get_catalog_summary         = _mod.get_catalog_summary
LLM_MODEL                 = _mod.LLM_MODEL
COLLECTION_NAME          = _mod.COLLECTION_NAME
QDRANT_HOST              = _mod.QDRANT_HOST
QDRANT_PORT              = _mod.QDRANT_PORT
QUERY_PREFIX             = _mod.QUERY_PREFIX
SYSTEM_PROMPT            = _mod.SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TOP_K_RETRIEVE    = 20      # fetch more from Qdrant, MMR narrows to top 3
TOP_N_FINAL       = 3       # final recommendations shown to user
MMR_LAMBDA        = 0.7     # 0 = max diversity, 1 = max relevance
HYBRID_COLLECTION = "products_hybrid"
BM25_MODEL        = "Qdrant/bm25"


# ── Chat session ──────────────────────────────────────────────────────────────
@dataclass
class Turn:
    role: str      # "user" or "assistant"
    content: str


@dataclass
class ChatSession:
    history: list[Turn]  = field(default_factory=list)
    preferences: dict    = field(default_factory=dict)  # max_price, min_rating, brand

    def add(self, role: str, content: str):
        self.history.append(Turn(role=role, content=content))

    def history_text(self, n_exchanges: int = 3) -> str:
        recent = self.history[-(n_exchanges * 2):]
        return "\n".join(f"{t.role.capitalize()}: {t.content}" for t in recent)

    def clear(self):
        self.history.clear()
        self.preferences.clear()
        log.info("  Session cleared.")


# ── Query rewriting ───────────────────────────────────────────────────────────
def rewrite_query(user_message: str, session: ChatSession, llm_client) -> str:
    if not session.history:
        return user_message

    prompt = (
        "Given the conversation history and a follow-up question, "
        "rewrite the follow-up into a complete standalone search query. "
        "Preserve any price limits, brand names, or product types from the history if still relevant.\n\n"
        f"Conversation:\n{session.history_text(n_exchanges=2)}\n\n"
        f"Follow-up: {user_message}\n\n"
        "Standalone search query (output only the query, nothing else):"
    )
    try:
        stream = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        rewritten = "".join(chunks).strip()
    except Exception as e:
        log.warning(f"  Query rewrite failed ({e}), using original message")
        return user_message
    if rewritten != user_message:
        log.info(f"  Rewritten: '{user_message}' → '{rewritten}'")
    return rewritten


# ── MMR re-ranking ────────────────────────────────────────────────────────────
def mmr_rerank(query_vector: list, scored_points: list, top_n: int, lam: float) -> list:
    """Maximal Marginal Relevance — balances relevance with diversity."""
    if len(scored_points) <= top_n:
        return scored_points

    vecs     = [np.array(p.vector) for p in scored_points]
    selected = []
    candidates = list(range(len(scored_points)))

    while len(selected) < top_n and candidates:
        best_score, best_idx = -float("inf"), None
        for idx in candidates:
            relevance = scored_points[idx].score
            if not selected:
                mmr = relevance
            else:
                # vectors are L2-normalised so dot product = cosine similarity
                max_sim = max(float(np.dot(vecs[idx], vecs[s])) for s in selected)
                mmr = lam * relevance - (1 - lam) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        candidates.remove(best_idx)

    return [scored_points[i] for i in selected]


# ── Retrieval with MMR ────────────────────────────────────────────────────────
def retrieve_diverse(
    query: str,
    embedder,
    client: QdrantClient,
    session: ChatSession,
    bm25_model: Optional[SparseTextEmbedding] = None,
    raw_message: Optional[str] = None,
) -> list[dict]:

    # Constraints come from the RAW user message's INTENT, not the rewrite.
    # The LLM rewrite invents numbers the user never said (it turned "better
    # ratings" into "under $600", anchoring on the previous result); acting on
    # those silently over-constrains the search. Rule: only change a constraint
    # if the raw message expresses that intent — using the rewrite ONLY to fill
    # in a number when the raw cue had none ("cheaper" → "$800"). And never
    # auto-loosen: price only lowers, rating only rises.
    raw = (raw_message or "").lower()

    # --- Price ---
    raw_price = extract_price_from_query(raw_message or "")
    price_cue = any(w in raw for w in (
        "cheap", "budget", "afford", "expensive", "pricier",
        "lower price", "less expensive", "spend less", "lower budget",
    ))
    if raw_price or price_cue:
        detected_price = raw_price or extract_price_from_query(query)
        current = session.preferences.get("max_price")
        if detected_price and (current is None or detected_price < current):
            session.preferences["max_price"] = detected_price
            log.info(f"  Price tightened to ${detected_price:.0f}")
        elif detected_price:
            log.info(f"  Ignoring higher price ${detected_price:.0f} (keeping ${current:.0f})")

    # --- Rating ---
    detected_rating = extract_rating_from_query(raw_message or "")
    if detected_rating is None and any(w in raw for w in ("rating", "rated", "star", "review")):
        detected_rating = extract_rating_from_query(query)   # raw asked, borrow the number
    if detected_rating:
        current = session.preferences.get("min_rating")
        if current is None or detected_rating > current:
            session.preferences["min_rating"] = detected_rating
            log.info(f"  Min rating tightened to {detected_rating}")

    max_price  = session.preferences.get("max_price")
    min_rating = session.preferences.get("min_rating")
    brand      = session.preferences.get("brand")

    # Category: detect per-turn from raw message (not persisted — users switch product types)
    category = extract_category_from_query(raw_message or query)
    if category:
        log.info(f"  Category filter: {category}")

    dense_vector = embedder.encode(
        QUERY_PREFIX + query, normalize_embeddings=True
    ).tolist()

    conditions = []
    if max_price:
        conditions.append(FieldCondition(key="price_numeric", range=Range(gte=0.01, lte=max_price)))
    if min_rating:
        conditions.append(FieldCondition(key="average_rating", range=Range(gte=min_rating)))
    if brand:
        conditions.append(FieldCondition(key="brand", match=MatchValue(value=brand)))
    if category:
        conditions.append(FieldCondition(key="main_category", match=MatchValue(value=category)))

    query_filter = Filter(must=conditions) if conditions else None

    if bm25_model is not None:
        # Hybrid search — dense + BM25 with RRF fusion
        sparse_emb = list(bm25_model.embed([query]))[0]
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
        results = client.query_points(
            collection_name=HYBRID_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector, using="dense",  limit=TOP_K_RETRIEVE),
                Prefetch(query=sparse_vec,   using="sparse", limit=TOP_K_RETRIEVE),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=TOP_K_RETRIEVE,
            with_payload=True,
            with_vectors=["dense"],   # only dense needed for MMR
        )
        log.info("  [hybrid search]")
    else:
        # Dense-only search
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vector,
            query_filter=query_filter,
            limit=TOP_K_RETRIEVE,
            with_payload=True,
            with_vectors=True,
        )

    # For MMR: extract dense vectors (handles both named and unnamed vector format)
    def get_dense_vec(p):
        if isinstance(p.vector, dict):
            return p.vector.get("dense", [])
        return p.vector or []

    # Patch vector attribute for MMR (expects p.vector as list)
    for p in results.points:
        p.vector = get_dense_vec(p)

    diverse = mmr_rerank(dense_vector, results.points, top_n=TOP_N_FINAL, lam=MMR_LAMBDA)

    return [
        {
            "title":        p.payload.get("title", ""),
            "brand":        p.payload.get("brand", ""),
            "price":        p.payload.get("price_numeric"),
            "rating":       p.payload.get("average_rating"),
            "rating_count": p.payload.get("rating_number", 0),
            "category":     p.payload.get("main_category", ""),
            "sub_category": p.payload.get("sub_category", ""),
            "score":        round(p.score, 4),
        }
        for p in diverse
    ]


# ── Conversational generation ─────────────────────────────────────────────────
def generate_with_history(
    user_message: str,
    context: str,
    session: ChatSession,
    llm_client,
) -> str:
    history_block = ""
    if session.history:
        history_block = f"Conversation so far:\n{session.history_text(n_exchanges=3)}\n\n"

    prompt = (
        f"{history_block}"
        f"Current request: {user_message}\n\n"
        f"Retrieved products:\n{context}\n\n"
        f"Provide your recommendation:"
    )
    stream = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )
    chunks = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    return "".join(chunks)


# ── Greeting detection ────────────────────────────────────────────────────────
_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "what's up", "whats up", "sup", "yo",
}

_GREETING_RESPONSE = (
    "Hello! I'm your electronics shopping assistant. "
    "I can help you find laptops, headphones, monitors, cameras, and much more "
    "from a catalog of 200K+ products.\n\n"
    "Just tell me what you're looking for — for example:\n"
    "- *'Gaming laptop under $800'*\n"
    "- *'Wireless headphones with good ratings'*\n"
    "- *'Sony camera under $500'*"
)


def is_greeting(text: str) -> bool:
    t = text.lower().strip().rstrip("!.,?")
    return t in _GREETINGS


# ── One conversation turn ─────────────────────────────────────────────────────
def chat_turn(
    user_message: str,
    session: ChatSession,
    embedder,
    qdrant: QdrantClient,
    llm_client,
    bm25_model: Optional[SparseTextEmbedding] = None,
) -> dict:
    if is_greeting(user_message):
        session.add("user",      user_message)
        session.add("assistant", _GREETING_RESPONSE)
        return {"search_query": user_message, "products": [], "response": _GREETING_RESPONSE}

    if is_capability_query(user_message):
        response = get_catalog_summary(qdrant)
        session.add("user",      user_message)
        session.add("assistant", response)
        return {"search_query": user_message, "products": [], "response": response}

    search_query = rewrite_query(user_message, session, llm_client)
    products     = retrieve_diverse(search_query, embedder, qdrant, session, bm25_model, raw_message=user_message)
    context      = format_context(products)
    response     = generate_with_history(user_message, context, session, llm_client)

    session.add("user",      user_message)
    session.add("assistant", response)

    return {"search_query": search_query, "products": products, "response": response}


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_float(prompt_text: str) -> Optional[float]:
    val = input(prompt_text).strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        print("  Invalid — skipping.")
        return None


def main():
    use_hybrid = "--hybrid" in sys.argv

    log.info("=== Initialising Conversational RAG ===")
    embedder = load_embedder()
    qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    llm      = load_llm()

    bm25 = None
    if use_hybrid:
        log.info("  Loading BM25 sparse model...")
        bm25 = SparseTextEmbedding(model_name=BM25_MODEL)
        count = qdrant.count(HYBRID_COLLECTION).count
        log.info(f"  Hybrid Qdrant ready — {count:,} products")
    else:
        count = qdrant.count(COLLECTION_NAME).count
        log.info(f"  Qdrant ready — {count:,} products indexed")

    session = ChatSession()
    mode_label = "Hybrid (Dense + BM25)" if use_hybrid else "Dense only"

    print("\n" + "=" * 60)
    print(f"  E-Commerce Conversational Assistant  [{mode_label}]")
    print("  'clear' — reset session  |  'quit' — exit")
    print("=" * 60)

    first_turn = True

    while True:
        print()
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            session.clear()
            first_turn = True
            print("  Session cleared — start a new conversation.")
            continue

        # Ask for persistent session filters only on the first real product query.
        # Skip the prompts entirely for capability questions ("what can you help me with?")
        # so the catalog summary is shown immediately without interruption.
        if first_turn and not is_capability_query(user_input) and not is_greeting(user_input):
            price = _parse_float("Max price in $ (Enter to skip): ")
            if price:
                session.preferences["max_price"] = price
                print(f"  Price limit ${price:.0f} saved for this session.")
            rating = _parse_float("Min rating e.g. 4.0 (Enter to skip): ")
            if rating:
                session.preferences["min_rating"] = rating
                print(f"  Min rating {rating} saved for this session.")
            first_turn = False
        elif first_turn and (is_capability_query(user_input) or is_greeting(user_input)):
            pass  # keep first_turn=True so filters are still asked on the next real query

        print("\nThinking...\n")
        start   = time.time()
        result  = chat_turn(user_input, session, embedder, qdrant, llm, bm25)
        elapsed = round(time.time() - start, 2)

        print(f"Assistant:\n{result['response']}")

        if result["search_query"] != user_input:
            print(f"\n  [Rewritten query: {result['search_query']}]")

        print(f"  [{elapsed}s | {len(result['products'])} products | "
              f"Top score: {result['products'][0]['score'] if result['products'] else 'N/A'}]")
        print("-" * 60)


if __name__ == "__main__":
    main()
