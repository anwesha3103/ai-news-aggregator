from backend.news_fetcher import fetch_and_store, get_articles_by_category, search_articles
from backend.summarizer import summarize_article, translate_text, get_or_create_summary, load_summarizer
from backend.recommender import (
    get_or_create_user, set_user_preferences,
    record_article_view, recommend_similar_articles, get_personalised_feed
)
