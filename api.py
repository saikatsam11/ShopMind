"""
Phase 8 — FastAPI Backend
Wraps the conversational RAG pipeline as a REST API.

Endpoints:
    GET  /health          — liveness check
    POST /chat            — send a message, get a recommendation
    POST /session/clear   — reset session history + preferences
    GET  /catalog         — live catalog summary

Run locally:
    uvicorn api:app --reload --port 8000
"""

import importlib.util
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ── Import pipeline modules ────────────────────────────────────────────────────
def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_rag   = _load_module("rag_pipeline",      "rag_pipeline.py")
_conv  = _load_module("conversational_rag", "conversational_rag.py")

# ── App state ──────────────────────────────────────────────────────────────────
class AppState:
    embedder = None
    qdrant   = None
    llm      = None
    sessions: dict = {}

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from qdrant_client import QdrantClient
    print("Loading embedder...")
    state.embedder = _rag.load_embedder()
    print("Connecting to Qdrant...")
    state.qdrant   = QdrantClient(
        host=os.getenv("QDRANT_HOST", _rag.QDRANT_HOST),
        port=int(os.getenv("QDRANT_PORT", _rag.QDRANT_PORT)),
        timeout=60,
    )
    print("Loading LLM...")
    state.llm = _rag.load_llm()
    print("API ready.")
    yield
    state.sessions.clear()


app = FastAPI(
    title="ShopMind API",
    description="Conversational product recommendation API — ShopMind backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None


class ProductInfo(BaseModel):
    title:        str
    brand:        str
    price:        Optional[float]
    rating:       Optional[float]
    rating_count: int
    category:     str
    score:        float


class ChatResponse(BaseModel):
    session_id:   str
    response:     str
    products:     list[ProductInfo]
    search_query: str


class ClearRequest(BaseModel):
    session_id: str


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_or_create_session(session_id: Optional[str]) -> tuple[str, object]:
    if not session_id or session_id.strip().lower() == "null":
        session_id = None
    if not session_id or session_id not in state.sessions:
        session_id = str(uuid.uuid4())
        state.sessions[session_id] = _conv.ChatSession()
    return session_id, state.sessions[session_id]


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    count = state.qdrant.count(_rag.COLLECTION_NAME).count if state.qdrant else 0
    return {"status": "ok", "products_indexed": count}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id, session = _get_or_create_session(req.session_id)

    result = _conv.chat_turn(
        user_message=req.message,
        session=session,
        embedder=state.embedder,
        qdrant=state.qdrant,
        llm_client=state.llm,
    )

    products = [
        ProductInfo(
            title=p.get("title", ""),
            brand=p.get("brand", ""),
            price=p.get("price"),
            rating=p.get("rating"),
            rating_count=p.get("rating_count", 0),
            category=p.get("sub_category") or p.get("category", ""),
            score=p.get("score", 0.0),
        )
        for p in result.get("products", [])
        if p.get("score", 0.0) >= 0.74
    ]

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        products=products,
        search_query=result["search_query"],
    )


@app.post("/session/clear")
def clear_session(req: ClearRequest):
    if req.session_id in state.sessions:
        state.sessions[req.session_id].clear()
    return {"status": "cleared", "session_id": req.session_id}


@app.get("/catalog")
def catalog():
    summary = _rag.get_catalog_summary(state.qdrant)
    return {"summary": summary}
