# config/settings.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the AI News Aggregator.
# All secrets are pulled from environment variables (see .env.example).
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()

# ── NewsAPI ───────────────────────────────────────────────────────────────────
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
NEWS_API_BASE_URL: str = "https://newsapi.org/v2"

# Default categories to fetch on startup / scheduled refresh
DEFAULT_CATEGORIES: list[str] = [
    "technology", "science", "business", "health", "entertainment", "sports"
]

# How many articles to fetch per category per refresh cycle
ARTICLES_PER_CATEGORY: int = 10

# ── PostgreSQL ────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/news_aggregator"
)

# ── T5 Summarisation ──────────────────────────────────────────────────────────
# Options: "t5-small" | "t5-base" | "t5-large"
# t5-small is fast for local dev; swap to t5-base for better quality
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "t5-small")

# Summarisation length controls
SUMMARY_MIN_LENGTH: int = 40
SUMMARY_MAX_LENGTH: int = 150

# ── Translation ───────────────────────────────────────────────────────────────
# Helsinki-NLP MarianMT models are used for translation.
# Format: "Helsinki-NLP/opus-mt-{src}-{tgt}"
SUPPORTED_LANGUAGES: dict[str, str] = {
    "French":   "Helsinki-NLP/opus-mt-en-fr",
    "Spanish":  "Helsinki-NLP/opus-mt-en-es",
    "German":   "Helsinki-NLP/opus-mt-en-de",
    "Hindi":    "Helsinki-NLP/opus-mt-en-hi",
    "Japanese": "Helsinki-NLP/opus-mt-en-jap",
}

# ── Recommendation Engine ─────────────────────────────────────────────────────
# TF-IDF cosine similarity threshold for "related articles"
SIMILARITY_THRESHOLD: float = 0.15

# Max recommended articles shown per article
MAX_RECOMMENDATIONS: int = 5

# ── Streamlit App ─────────────────────────────────────────────────────────────
APP_TITLE: str = "AI News Aggregator"
APP_ICON: str = "🗞️"
ARTICLES_PER_PAGE: int = 9
