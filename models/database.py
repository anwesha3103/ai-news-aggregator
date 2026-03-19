# models/database.py
# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy ORM models + engine / session setup.
#
# Tables:
#   articles   – raw fetched articles (title, content, url, source, category…)
#   summaries  – T5-generated summaries linked to articles
#   users      – lightweight user preference store
#   user_prefs – many-to-many: which categories a user follows
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, Float, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from config.settings import DATABASE_URL

# ── Engine & Session ──────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # silently reconnect on stale connections
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class Article(Base):
    """
    Stores a single news article fetched from NewsAPI.
    The `url` field acts as a natural unique key to prevent duplicates.
    """
    __tablename__ = "articles"

    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)       # short blurb from NewsAPI
    content     = Column(Text, nullable=True)       # full body (may be truncated by NewsAPI)
    url         = Column(String(1000), unique=True, nullable=False)
    image_url   = Column(String(1000), nullable=True)
    source      = Column(String(200), nullable=True)
    author      = Column(String(200), nullable=True)
    category    = Column(String(100), nullable=False, index=True)
    published_at= Column(DateTime, nullable=True)
    fetched_at  = Column(DateTime, default=datetime.utcnow)

    # Relationship: one article → one summary (optional)
    summary = relationship("Summary", back_populates="article", uselist=False)

    def __repr__(self):
        return f"<Article id={self.id} category={self.category!r} title={self.title[:40]!r}>"


class Summary(Base):
    """
    T5-generated summary for an Article, with optional translation.
    Multiple translated rows can exist per article (one per language).
    """
    __tablename__ = "summaries"
    __table_args__ = (
        UniqueConstraint("article_id", "language", name="uq_summary_lang"),
    )

    id          = Column(Integer, primary_key=True, index=True)
    article_id  = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    language    = Column(String(50), default="English", nullable=False)
    summary_text= Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)

    article = relationship("Article", back_populates="summary")

    def __repr__(self):
        return f"<Summary article_id={self.article_id} lang={self.language!r}>"


class User(Base):
    """
    Minimal user record – stores display name + preferred categories.
    In a production app you'd add auth; here it's kept simple for the demo.
    """
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r}>"


class UserPreference(Base):
    """
    Tracks which categories a user has opted into and
    stores a simple engagement score (incremented on article views).
    """
    __tablename__ = "user_preferences"
    __table_args__ = (
        UniqueConstraint("user_id", "category", name="uq_user_category"),
    )

    id        = Column(Integer, primary_key=True, index=True)
    user_id   = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    category  = Column(String(100), nullable=False)
    score     = Column(Float, default=1.0)   # higher = stronger interest

    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreference user={self.user_id} category={self.category!r} score={self.score}>"


# ── Helpers ───────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created / verified.")


def get_db():
    """
    FastAPI / script dependency that yields a DB session
    and guarantees it is closed afterwards.

    Usage (FastAPI):
        @app.get("/articles")
        def list_articles(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
