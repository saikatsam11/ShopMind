"""
Phase 9 — Streamlit Frontend
Conversational UI for the E-Commerce RAG Assistant.

Run:
    streamlit run streamlit_app.py
Make sure FastAPI is running first:
    uvicorn api:app --reload --port 8000
"""

import requests
import streamlit as st

API_BASE = "https://saix11-shopmind-api.hf.space"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopMind",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background — deep blue-purple gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 40%, #1a0a2e 100%);
        min-height: 100vh;
    }

    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #130d2e 100%) !important;
        border-right: 1px solid #1e2a4a !important;
    }

    /* Subtle glow on header */
    h1, h2, h3 { text-shadow: 0 0 30px rgba(100, 120, 255, 0.3); }

    /* Chat input glows on focus */
    .stChatInput textarea:focus {
        border-color: #4f6ef7 !important;
        box-shadow: 0 0 12px rgba(79, 110, 247, 0.4) !important;
    }

    /* Divider color */
    hr { border-color: #1e2a4a !important; }

    /* Product card */
    .product-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(100, 120, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: border-color 0.2s, box-shadow 0.2s;
        backdrop-filter: blur(6px);
    }
    .product-card:hover {
        border-color: #4f8ef7;
        box-shadow: 0 0 16px rgba(79, 142, 247, 0.2);
    }

    .product-title {
        font-size: 14px;
        font-weight: 600;
        color: #e8eaf6;
        line-height: 1.4;
        margin-bottom: 8px;
    }
    .product-brand {
        font-size: 12px;
        color: #7986cb;
        margin-bottom: 6px;
    }
    .product-meta {
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
    }
    .price-tag {
        font-size: 16px;
        font-weight: 700;
        color: #69f0ae;
    }
    .rating-tag {
        font-size: 13px;
        color: #ffd54f;
    }
    .category-tag {
        font-size: 11px;
        background: #2d3250;
        color: #9fa8da;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .match-tag {
        font-size: 11px;
        color: #80cbc4;
        margin-left: auto;
    }

    /* Search query badge */
    .search-badge {
        background: rgba(26, 35, 126, 0.5);
        border: 1px solid rgba(79, 110, 247, 0.4);
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 13px;
        color: #9fa8da;
        margin-bottom: 14px;
        backdrop-filter: blur(4px);
    }

    /* Status bar */
    .status-bar {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(100,120,255,0.15);
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 12px;
        color: #546e7a;
        text-align: center;
        margin-top: 10px;
    }

    /* Header */
    .main-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0 20px 0;
    }

    /* Hide default streamlit header padding */
    .block-container { padding-top: 1.5rem; }

    /* Chat input placeholder */
    .stChatInput textarea { background: #1e2130 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "session_id"    not in st.session_state: st.session_state.session_id    = None
if "messages"      not in st.session_state: st.session_state.messages      = []
if "last_products" not in st.session_state: st.session_state.last_products = []
if "last_query"    not in st.session_state: st.session_state.last_query    = ""
if "api_ok"        not in st.session_state: st.session_state.api_ok        = None
if "catalog"       not in st.session_state: st.session_state.catalog       = None


# ── API helpers ────────────────────────────────────────────────────────────────
def check_api() -> bool:
    for _ in range(3):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=30)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


def send_message(message: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"message": message, "session_id": st.session_state.session_id},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def fetch_catalog() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/catalog", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def clear_session():
    if st.session_state.session_id:
        try:
            requests.post(
                f"{API_BASE}/session/clear",
                json={"session_id": st.session_state.session_id},
                timeout=5,
            )
        except Exception:
            pass
    st.session_state.session_id    = None
    st.session_state.messages      = []
    st.session_state.last_products = []
    st.session_state.last_query    = ""


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_btn = st.columns([5, 1])
with col_title:
    st.markdown("## 🛒 ShopMind")
    st.caption("Your AI-powered personal shopping assistant")
with col_btn:
    st.write("")
    st.write("")
    if st.button("Clear Session", use_container_width=True):
        clear_session()
        st.rerun()

st.divider()

# ── Check API health once per session ─────────────────────────────────────────
if st.session_state.api_ok is None:
    st.session_state.api_ok = check_api()

if not st.session_state.api_ok:
    st.warning("Connecting to ShopMind API...")
    if st.button("Retry Connection"):
        st.session_state.api_ok = None
        st.rerun()
    st.stop()

# ── Sidebar: Catalog browser ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Catalog")

    if st.button("Refresh Catalog", use_container_width=True):
        st.session_state.catalog = fetch_catalog()

    if st.session_state.catalog is None:
        st.session_state.catalog = fetch_catalog()

    if st.session_state.catalog:
        summary_text = st.session_state.catalog.get("summary", "").replace("$", "\\$")
        st.markdown(summary_text)
    else:
        st.warning("Could not load catalog.")

    st.divider()
    st.markdown("**Quick prompts — click to copy:**")
    prompts = [
        "Gaming laptop under $800",
        "Best wireless earbuds under $100",
        "4K mirrorless camera under $1000",
        "Mechanical keyboard under $150",
        "Smartwatch with heart rate monitor",
        "Budget webcam for video calls",
        "External SSD under $80",
        "Bluetooth speaker under $50",
    ]
    for p in prompts:
        st.code(p, language=None)

# ── Main layout ────────────────────────────────────────────────────────────────
chat_col, prod_col = st.columns([55, 45], gap="large")

# ── LEFT: Chat panel ───────────────────────────────────────────────────────────
with chat_col:
    st.markdown("### Chat")

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Welcome message on first load
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "Hi! I'm **ShopMind**, your AI shopping assistant. Tell me what you're "
                "looking for — budget, brand, category, or any specs — and I'll find "
                "the best matches from our catalog.\n\n"
                "**Try:** *\"Gaming laptop under $800\"* or *\"Best wireless earbuds for gym\"*"
            )

    # Chat input
    user_input = st.chat_input("Ask about any product...")

    if user_input:
        # Show user bubble immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call API
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching catalog..."):
                result = send_message(user_input)

            if result is None:
                response_text = "Sorry, I couldn't reach the backend. Please check that FastAPI is running."
                st.session_state.api_ok = False
            else:
                response_text = result["response"]
                st.session_state.session_id    = result["session_id"]
                st.session_state.last_products = result.get("products", [])
                st.session_state.last_query    = result.get("search_query", "")

            st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()

# ── RIGHT: Products panel ──────────────────────────────────────────────────────
with prod_col:
    st.markdown("### Products Found")

    if st.session_state.last_query:
        st.markdown(
            f'<div class="search-badge">🔍 &nbsp;<b>Search:</b> {st.session_state.last_query}</div>',
            unsafe_allow_html=True,
        )

    if not st.session_state.last_products:
        st.markdown(
            "<div style='color:#546e7a; margin-top:40px; text-align:center;'>"
            "Product results will appear here after your first message."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for p in st.session_state.last_products:
            price_str  = f"${p['price']:.2f}"  if p.get("price")  else "N/A"
            rating_str = f"⭐ {p['rating']:.1f}" if p.get("rating") else "No rating"
            match_pct  = f"{p.get('score', 0) * 100:.1f}%"
            title      = p.get("title", "Unknown")
            brand      = p.get("brand", "")
            category   = p.get("category", "")
            count      = p.get("rating_count", 0)

            st.markdown(f"""
<div class="product-card">
    <div class="product-title">{title}</div>
    <div class="product-brand">{brand}</div>
    <div class="product-meta">
        <span class="price-tag">{price_str}</span>
        <span class="rating-tag">{rating_str}</span>
        <span style="font-size:11px; color:#546e7a;">({count} reviews)</span>
        <span class="category-tag">{category}</span>
        <span class="match-tag">Match {match_pct}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Status bar
    session_label = f"Session: `{st.session_state.session_id[:8]}...`" if st.session_state.session_id else "No active session"
    product_count = len(st.session_state.last_products)
    st.markdown(
        f'<div class="status-bar">Products: {product_count} &nbsp;|&nbsp; {session_label} &nbsp;|&nbsp; API: online</div>',
        unsafe_allow_html=True,
    )
