# backend/news_fetcher.py
# ─────────────────────────────────────────────────────────────────────────────
# Fetches articles from NewsAPI and persists them to PostgreSQL.
#
# Key design decisions
# ─────────────────────
# • Uses the /v2/top-headlines endpoint (category-level granularity).
# • De-duplicates by URL before inserting – safe to call repeatedly.
# • Returns a list of Article ORM objects so callers can chain operations.
# ─────────────────────────────────────────────────────────────────────────────

import requests
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from config.settings import (
    NEWS_API_KEY, NEWS_API_BASE_URL,
    DEFAULT_CATEGORIES, ARTICLES_PER_CATEGORY,
)
from models.database import Article


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> datetime | None:
    """Convert NewsAPI ISO-8601 string to a Python datetime (UTC)."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def _fetch_headlines(category: str, page_size: int = ARTICLES_PER_CATEGORY) -> list[dict]:
    """
    Hit the NewsAPI /top-headlines endpoint for a single category.
    Returns the raw list of article dicts (or [] on error).
    """
    if not NEWS_API_KEY:
        raise EnvironmentError(
            "NEWS_API_KEY not set. Add it to your .env file."
        )

    params = {
        "apiKey":   NEWS_API_KEY,
        "category": category,
        "language": "en",
        "pageSize": page_size,
    }

    try:
        resp = requests.get(
            f"{NEWS_API_BASE_URL}/top-headlines",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            print(f"⚠️  NewsAPI error for '{category}': {data.get('message')}")
            return []

        return data.get("articles", [])

    except requests.RequestException as exc:
        print(f"❌ Network error fetching '{category}': {exc}")
        return []


def _article_exists(db: Session, url: str) -> bool:
    """Return True if an article with this URL is already in the DB."""
    return db.query(Article).filter(Article.url == url).first() is not None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_and_store(
    db: Session,
    categories: list[str] | None = None,
) -> list[Article]:
    """
    Fetch headlines for every category in `categories` (defaults to
    DEFAULT_CATEGORIES), persist new articles to the DB, and return all
    newly inserted Article objects.

    Parameters
    ----------
    db         : Active SQLAlchemy session.
    categories : List of category strings. Pass None to use defaults.

    Returns
    -------
    list[Article] – only the articles inserted in this call (not duplicates).
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES

    inserted: list[Article] = []

    for cat in categories:
        raw_articles = _fetch_headlines(cat)
        print(f"📡 Fetched {len(raw_articles)} articles for '{cat}'")

        for raw in raw_articles:
            url = raw.get("url", "").strip()
            if not url or url == "https://removed.com":
                continue                          # skip placeholder / removed articles

            if _article_exists(db, url):
                continue                          # de-duplicate

            article = Article(
                title        = raw.get("title", "Untitled")[:500],
                description  = raw.get("description"),
                content      = raw.get("content"),
                url          = url,
                image_url    = raw.get("urlToImage"),
                source       = raw.get("source", {}).get("name"),
                author       = raw.get("author"),
                category     = cat,
                published_at = _parse_date(raw.get("publishedAt")),
            )
            db.add(article)

            try:
                db.commit()
                db.refresh(article)
                inserted.append(article)
            except IntegrityError:
                # Race condition: another process inserted the same URL
                db.rollback()

    print(f"✅ Inserted {len(inserted)} new articles across {len(categories)} categories.")
    return inserted


def get_articles_by_category(
    db: Session,
    category: str,
    limit: int = 20,
    offset: int = 0,
) -> list[Article]:
    """
    Retrieve stored articles for a given category, newest first.
    Supports pagination via limit / offset.
    """
    return (
        db.query(Article)
        .filter(Article.category == category)
        .order_by(Article.published_at.desc().nullslast())
        .offset(offset)
        .limit(limit)
        .all()
    )


def search_articles(
    db: Session,
    query: str,
    limit: int = 20,
) -> list[Article]:
    """
    Simple full-text search across title and description.
    PostgreSQL ILIKE is used for case-insensitive matching.
    For production, swap this for a proper tsvector index.
    """
    pattern = f"%{query}%"
    return (
        db.query(Article)
        .filter(
            Article.title.ilike(pattern) |
            Article.description.ilike(pattern)
        )
        .order_by(Article.published_at.desc().nullslast())
        .limit(limit)
        .all()
    )
