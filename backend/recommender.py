# backend/recommender.py
# ─────────────────────────────────────────────────────────────────────────────
# Personalised article recommendation engine.
#
# Two complementary strategies
# ────────────────────────────
# 1. Category-based filtering
#    Pull articles from categories the user has expressed interest in,
#    weighted by their engagement score (UserPreference.score).
#
# 2. Content-based TF-IDF similarity
#    Given a seed article, find the most similar articles in the DB
#    using cosine similarity on TF-IDF vectors built from title + description.
#
# This mirrors what production recommenders do before introducing
# collaborative filtering – a solid baseline for a portfolio project.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
import numpy as np

from config.settings import SIMILARITY_THRESHOLD, MAX_RECOMMENDATIONS
from models.database import Article, User, UserPreference


# ── User preference helpers ───────────────────────────────────────────────────

def get_or_create_user(db: Session, username: str) -> User:
    """Return an existing User or create a new one."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def set_user_preferences(
    db: Session,
    user: User,
    categories: list[str],
) -> None:
    """
    Replace the user's category preferences with the given list.
    Existing preferences not in the new list are deleted; new ones are added.
    """
    # Build a quick lookup of current preferences
    current = {p.category: p for p in user.preferences}

    # Remove categories no longer wanted
    for cat, pref in current.items():
        if cat not in categories:
            db.delete(pref)

    # Add new categories (score defaults to 1.0)
    for cat in categories:
        if cat not in current:
            db.add(UserPreference(user_id=user.id, category=cat, score=1.0))

    db.commit()


def record_article_view(db: Session, user: User, article: Article) -> None:
    """
    Increment the engagement score for the article's category.
    Called whenever a user expands / reads an article in the UI.
    """
    pref = (
        db.query(UserPreference)
        .filter(UserPreference.user_id == user.id, UserPreference.category == article.category)
        .first()
    )
    if pref:
        pref.score += 0.5          # incremental weight for each view
    else:
        # Auto-add category if user views an article outside their explicit prefs
        db.add(UserPreference(user_id=user.id, category=article.category, score=1.5))
    db.commit()


# ── Recommendation strategies ─────────────────────────────────────────────────

def recommend_by_preferences(
    db: Session,
    user: User,
    limit: int = MAX_RECOMMENDATIONS * 2,
) -> list[Article]:
    """
    Return articles from the user's preferred categories,
    ordering by (preference score DESC, published_at DESC).

    Falls back to latest articles across all categories if the user
    has no stored preferences yet.
    """
    prefs = user.preferences
    if not prefs:
        # Cold start: return most recent articles
        return (
            db.query(Article)
            .order_by(Article.published_at.desc().nullslast())
            .limit(limit)
            .all()
        )

    # Sort preferences by score descending
    sorted_prefs = sorted(prefs, key=lambda p: p.score, reverse=True)
    categories   = [p.category for p in sorted_prefs]

    articles: list[Article] = []
    per_cat = max(1, limit // len(categories))

    for cat in categories:
        batch = (
            db.query(Article)
            .filter(Article.category == cat)
            .order_by(Article.published_at.desc().nullslast())
            .limit(per_cat)
            .all()
        )
        articles.extend(batch)
        if len(articles) >= limit:
            break

    return articles[:limit]


def recommend_similar_articles(
    db: Session,
    seed_article: Article,
    limit: int = MAX_RECOMMENDATIONS,
) -> list[Article]:
    """
    Find articles similar to `seed_article` using TF-IDF cosine similarity.

    Steps:
      1. Pull all articles from the same category (manageable corpus).
      2. Build a TF-IDF matrix from title + description text.
      3. Compute cosine similarity of the seed against the corpus.
      4. Return the top-N most similar articles (excluding the seed itself).
    """
    # Fetch candidate pool (same category, recent)
    candidates: list[Article] = (
        db.query(Article)
        .filter(Article.category == seed_article.category,
                Article.id != seed_article.id)
        .order_by(Article.published_at.desc().nullslast())
        .limit(200)    # cap corpus size for performance
        .all()
    )

    if not candidates:
        return []

    # Build text corpus: [seed] + candidates
    def article_text(a: Article) -> str:
        return f"{a.title or ''} {a.description or ''}"

    corpus = [article_text(seed_article)] + [article_text(c) for c in candidates]

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Corpus might be all empty strings
        return []

    # Cosine similarity between seed (index 0) and all candidates
    similarities: np.ndarray = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Filter by threshold, then rank
    results: list[tuple[float, Article]] = [
        (sim, candidates[i])
        for i, sim in enumerate(similarities)
        if sim >= SIMILARITY_THRESHOLD
    ]
    results.sort(key=lambda x: x[0], reverse=True)

    return [art for _, art in results[:limit]]


def get_personalised_feed(
    db: Session,
    user: User,
    limit: int = 20,
) -> list[Article]:
    """
    Combine preference-based and recency signals into a single feed.
    This is what the Streamlit homepage renders.
    """
    pref_articles   = recommend_by_preferences(db, user, limit=limit)
    seen_ids        = {a.id for a in pref_articles}

    # Pad with recency if we got fewer than requested
    if len(pref_articles) < limit:
        recent = (
            db.query(Article)
            .filter(Article.id.notin_(seen_ids))
            .order_by(Article.published_at.desc().nullslast())
            .limit(limit - len(pref_articles))
            .all()
        )
        pref_articles.extend(recent)

    return pref_articles[:limit]
