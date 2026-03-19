# app.py  ←  run with:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit front-end for the AI News Aggregator.
#
# Pages / sections
# ─────────────────
#   🏠 Home         – personalised feed for the logged-in user
#   🔍 Search       – full-text search across stored articles
#   📂 Browse       – articles filtered by category
#   ⚙️  Settings    – update username + category preferences
#
# The sidebar also exposes a "Refresh News" button that triggers a fresh fetch
# from NewsAPI and runs the T5 summariser on all new articles.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
from sqlalchemy.orm import Session

from config.settings import (
    APP_TITLE, APP_ICON, DEFAULT_CATEGORIES,
    ARTICLES_PER_PAGE, SUPPORTED_LANGUAGES,
)
from models.database import SessionLocal, init_db, Article
from backend.news_fetcher import fetch_and_store, get_articles_by_category, search_articles
from backend.summarizer import get_or_create_summary, load_summarizer
from backend.recommender import (
    get_or_create_user, set_user_preferences,
    record_article_view, recommend_similar_articles, get_personalised_feed,
)

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure DB tables exist ────────────────────────────────────────────────────
init_db()

# ── Session state defaults ────────────────────────────────────────────────────
if "username"        not in st.session_state: st.session_state.username        = "guest"
if "selected_lang"   not in st.session_state: st.session_state.selected_lang   = "English"
if "active_page"     not in st.session_state: st.session_state.active_page     = "Home"
if "search_query"    not in st.session_state: st.session_state.search_query    = ""
if "browse_category" not in st.session_state: st.session_state.browse_category = DEFAULT_CATEGORIES[0]
if "model_loaded"    not in st.session_state:
    with st.spinner("Loading T5 summarisation model…"):
        load_summarizer()
    st.session_state.model_loaded = True


# ── DB helper ─────────────────────────────────────────────────────────────────
def get_db() -> Session:
    return SessionLocal()


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Card-style article blocks */
    .article-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .article-card:hover { border-color: #89b4fa; }

    .article-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #cdd6f4;
        text-decoration: none;
    }
    .article-meta {
        font-size: 0.78rem;
        color: #6c7086;
        margin-top: 0.25rem;
    }
    .article-summary {
        font-size: 0.88rem;
        color: #a6adc8;
        margin-top: 0.6rem;
        line-height: 1.55;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-right: 6px;
    }
    .badge-tech        { background:#313244; color:#89b4fa; }
    .badge-science     { background:#313244; color:#a6e3a1; }
    .badge-business    { background:#313244; color:#f9e2af; }
    .badge-health      { background:#313244; color:#f38ba8; }
    .badge-entertainment{ background:#313244; color:#cba6f7; }
    .badge-sports      { background:#313244; color:#fab387; }
    .badge-default     { background:#313244; color:#cdd6f4; }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] { background: #181825; }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: #313244;
        color: #cdd6f4;
        border: none;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #45475a;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: render a single article ──────────────────────────────────────────
BADGE_CLASSES = {
    "technology": "badge-tech", "science": "badge-science",
    "business": "badge-business", "health": "badge-health",
    "entertainment": "badge-entertainment", "sports": "badge-sports",
}

def render_article(article: Article, db: Session, user, language: str):
    """Render one article card with summary + related articles expander."""
    badge_cls = BADGE_CLASSES.get(article.category, "badge-default")
    pub_date  = article.published_at.strftime("%b %d, %Y") if article.published_at else "Unknown date"

    st.markdown(f"""
    <div class="article-card">
        <span class="badge {badge_cls}">{article.category}</span>
        <a class="article-title" href="{article.url}" target="_blank">{article.title}</a>
        <div class="article-meta">🗞️ {article.source or 'Unknown'} &nbsp;·&nbsp; 📅 {pub_date}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Read AI Summary"):
        record_article_view(db, user, article)   # track engagement
        with st.spinner("Generating summary…"):
            summary = get_or_create_summary(db, article, language)
        st.markdown(f'<p class="article-summary">{summary}</p>', unsafe_allow_html=True)
        st.caption(f"*Language: {language}*")

        # Similar articles
        similar = recommend_similar_articles(db, article, limit=3)
        if similar:
            st.markdown("**Related articles:**")
            for s in similar:
                st.markdown(f"- [{s.title}]({s.url})")

    st.markdown("<hr style='border-color:#313244;margin:0.5rem 0'>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.divider()

    # Navigation
    pages = {"🏠 Home": "Home", "🔍 Search": "Search", "📂 Browse": "Browse", "⚙️ Settings": "Settings"}
    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state.active_page = key

    st.divider()

    # Language selector
    lang_options = ["English"] + list(SUPPORTED_LANGUAGES.keys())
    st.session_state.selected_lang = st.selectbox(
        "🌍 Summary language", lang_options,
        index=lang_options.index(st.session_state.selected_lang)
    )

    st.divider()

    # Refresh news
    if st.button("🔄 Refresh News"):
        with st.spinner("Fetching latest articles…"):
            db = get_db()
            new_articles = fetch_and_store(db)
            db.close()
        st.success(f"✅ Added {len(new_articles)} new articles!")
        st.rerun()

    st.divider()
    st.caption(f"👤 Logged in as **{st.session_state.username}**")


# ── Main content ──────────────────────────────────────────────────────────────
page = st.session_state.active_page

# ──────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────
if page == "Home":
    st.title("🏠 Your Personalised Feed")
    db   = get_db()
    user = get_or_create_user(db, st.session_state.username)
    feed = get_personalised_feed(db, user, limit=ARTICLES_PER_PAGE)

    if not feed:
        st.info("No articles yet! Click **Refresh News** in the sidebar to fetch the latest headlines.")
    else:
        for article in feed:
            render_article(article, db, user, st.session_state.selected_lang)
    db.close()


# ──────────────────────────────────────────────
# SEARCH
# ──────────────────────────────────────────────
elif page == "Search":
    st.title("🔍 Search Articles")
    query = st.text_input("Enter keywords…", value=st.session_state.search_query, placeholder="e.g. artificial intelligence")
    st.session_state.search_query = query

    if query.strip():
        db      = get_db()
        user    = get_or_create_user(db, st.session_state.username)
        results = search_articles(db, query, limit=ARTICLES_PER_PAGE)
        st.caption(f"Found **{len(results)}** articles matching *{query}*")

        if results:
            for article in results:
                render_article(article, db, user, st.session_state.selected_lang)
        else:
            st.warning("No articles found. Try different keywords or refresh the news first.")
        db.close()


# ──────────────────────────────────────────────
# BROWSE
# ──────────────────────────────────────────────
elif page == "Browse":
    st.title("📂 Browse by Category")
    selected_cat = st.selectbox(
        "Category",
        DEFAULT_CATEGORIES,
        index=DEFAULT_CATEGORIES.index(st.session_state.browse_category)
    )
    st.session_state.browse_category = selected_cat

    db      = get_db()
    user    = get_or_create_user(db, st.session_state.username)
    articles = get_articles_by_category(db, selected_cat, limit=ARTICLES_PER_PAGE)
    st.caption(f"Showing latest **{len(articles)}** articles in *{selected_cat}*")

    if articles:
        for article in articles:
            render_article(article, db, user, st.session_state.selected_lang)
    else:
        st.info(f"No articles in '{selected_cat}' yet. Click **Refresh News** to fetch.")
    db.close()


# ──────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────
elif page == "Settings":
    st.title("⚙️ Settings")

    # Username
    st.subheader("👤 Profile")
    new_username = st.text_input("Username", value=st.session_state.username)
    if st.button("Update Username"):
        st.session_state.username = new_username
        st.success(f"Username updated to **{new_username}**!")
        st.rerun()

    st.divider()

    # Category preferences
    st.subheader("📌 Category Preferences")
    st.caption("Choose the topics you want to see in your personalised feed.")

    db   = get_db()
    user = get_or_create_user(db, st.session_state.username)

    current_cats = {p.category for p in user.preferences}
    selected_cats = st.multiselect(
        "Follow categories",
        options=DEFAULT_CATEGORIES,
        default=list(current_cats) if current_cats else DEFAULT_CATEGORIES[:3]
    )

    if st.button("💾 Save Preferences"):
        set_user_preferences(db, user, selected_cats)
        st.success("Preferences saved! Your feed will update on next visit.")

    db.close()

    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
    **AI News Aggregator** · built with:
    - 🗞️ [NewsAPI](https://newsapi.org) – live headline data
    - 🤗 [HuggingFace Transformers](https://huggingface.co) – T5 summarisation + MarianMT translation
    - 🐘 PostgreSQL – article & preference storage
    - 🎈 [Streamlit](https://streamlit.io) – interactive UI
    - 🔬 scikit-learn TF-IDF – content-based recommendations
    """)
