# backend/summarizer.py
# ─────────────────────────────────────────────────────────────────────────────
# Transformer-based summarisation using HuggingFace T5.
#
# Design notes
# ─────────────────────
# • Model is loaded ONCE at module level (singleton pattern) to avoid
#   reloading on every call – critical for Streamlit's re-run model.
# • Falls back gracefully to the article description if the content is
#   too short or the model has not been loaded yet.
# • The `translate_text()` function uses MarianMT models for translation,
#   downloading them lazily and caching them in a dict.
# ─────────────────────────────────────────────────────────────────────────────

import re
from functools import lru_cache
from sqlalchemy.orm import Session

from config.settings import (
    SUMMARIZATION_MODEL,
    SUMMARY_MIN_LENGTH, SUMMARY_MAX_LENGTH,
    SUPPORTED_LANGUAGES,
)
from models.database import Article, Summary

# Lazy imports – only pulled in when actually needed so the app starts fast
# even if the heavy HuggingFace libraries take a moment.
_summarizer = None          # T5 pipeline
_translators: dict = {}     # language → MarianMT pipeline


# ── Model loading ─────────────────────────────────────────────────────────────

def load_summarizer():
    """
    Load the T5 summarisation pipeline (once).
    Call this explicitly on app startup to warm the model.
    """
    global _summarizer
    if _summarizer is not None:
        return _summarizer

    from transformers import pipeline
    print(f" Loading summarisation model: {SUMMARIZATION_MODEL} …")
    _summarizer = pipeline(
    "text-generation",
    model=SUMMARIZATION_MODEL,
    tokenizer=SUMMARIZATION_MODEL,
    )
    print(" Summarisation model loaded.")
    return _summarizer


def load_translator(language: str):
    """
    Load a MarianMT translation pipeline for `language` (once, then cached).
    Language must be a key in SUPPORTED_LANGUAGES.
    """
    if language in _translators:
        return _translators[language]

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Choose from: {list(SUPPORTED_LANGUAGES)}"
        )

    from transformers import pipeline
    model_name = SUPPORTED_LANGUAGES[language]
    print(f" Loading translation model for {language}: {model_name} …")
    translator = pipeline("translation", model=model_name)
    _translators[language] = translator
    print(f" Translation model loaded for {language}.")
    return translator


# ── Text utilities ────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Strip NewsAPI content artifacts (e.g. ' [+1234 chars]') and
    normalise whitespace before feeding to T5.
    """
    text = re.sub(r"\[?\+?\d+\s*chars?\]?", "", text)   # remove char-count tags
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_input(article: Article) -> str | None:
    """
    Build the best possible input text for T5.
    Priority: full content > description > title.
    Returns None if there's nothing useful to summarise.
    """
    for field in (article.content, article.description, article.title):
        if field and len(field.strip()) > 30:
            return _clean_text(field)
    return None


# ── Core functions ────────────────────────────────────────────────────────────

def summarize_article(article: Article) -> str:
    """
    Generate a concise T5 summary for a single Article object.

    Returns the summary string (falls back to description/title if
    the model isn't loaded or text is too short).
    """
    summarizer = _summarizer or load_summarizer()

    input_text = _build_input(article)
    if not input_text:
        return article.description or article.title or "No content available."

    # T5 expects the "summarize: " prefix
    t5_input = f"summarize: {input_text}"

    try:
        result = summarizer(
            t5_input,
            max_length=SUMMARY_MAX_LENGTH,
            min_length=SUMMARY_MIN_LENGTH,
            do_sample=False,
            truncation=True,
        )
        return result[0]["generated_text"].strip()
    except Exception as exc:
        print(f"⚠️  Summarisation failed for article {article.id}: {exc}")
        return article.description or article.title or "Summary unavailable."


def translate_text(text: str, language: str) -> str:
    """
    Translate `text` from English to `language` using a MarianMT model.
    Returns the original text if translation fails.
    """
    if language == "English":
        return text

    try:
        translator = load_translator(language)
        result = translator(text, max_length=512, truncation=True)
        return result[0]["translation_text"].strip()
    except Exception as exc:
        print(f"⚠️  Translation to {language} failed: {exc}")
        return text


def summarize_and_store(
    db: Session,
    articles: list[Article],
    language: str = "English",
) -> list[Summary]:
    """
    Batch-summarise a list of articles and persist Summary rows to the DB.

    • Skips articles that already have a summary in the requested language.
    • Returns all Summary objects created in this call.

    Parameters
    ----------
    db       : Active SQLAlchemy session.
    articles : Articles to summarise (usually the batch returned by the fetcher).
    language : Target language for the summary text.
    """
    created: list[Summary] = []

    for article in articles:
        # Check whether a summary already exists for this language
        existing = (
            db.query(Summary)
            .filter(Summary.article_id == article.id, Summary.language == language)
            .first()
        )
        if existing:
            continue

        raw_summary = summarize_article(article)

        # Optionally translate
        if language != "English":
            raw_summary = translate_text(raw_summary, language)

        summary = Summary(
            article_id   = article.id,
            language     = language,
            summary_text = raw_summary,
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
        created.append(summary)
        print(f"📝 Summarised article {article.id} [{language}]")

    return created


def get_or_create_summary(
    db: Session,
    article: Article,
    language: str = "English",
) -> str:
    """
    Return the stored summary for an article in the requested language,
    generating and saving it on-the-fly if it doesn't exist yet.
    This is the main entry point called from the Streamlit UI.
    """
    existing = (
        db.query(Summary)
        .filter(Summary.article_id == article.id, Summary.language == language)
        .first()
    )
    if existing:
        return existing.summary_text

    # Generate fresh
    raw_summary = summarize_article(article)
    if language != "English":
        raw_summary = translate_text(raw_summary, language)

    summary = Summary(
        article_id   = article.id,
        language     = language,
        summary_text = raw_summary,
    )
    db.add(summary)
    db.commit()

    return raw_summary
