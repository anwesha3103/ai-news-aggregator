# 🗞️ AI-Based Personalized News Aggregator

An end-to-end AI news pipeline that fetches live headlines, generates concise transformer-based summaries, supports multi-language translation, and serves a personalised article feed — all through a clean Streamlit interface.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Live News Fetching** | Pulls top headlines from [NewsAPI](https://newsapi.org) across 6 categories |
| **T5 Summarisation** | HuggingFace `t5-small` (or `t5-base`) generates concise summaries |
| **Multi-language Translation** | MarianMT models translate summaries into French, Spanish, German, Hindi, and Japanese |
| **Personalised Feed** | Category preferences + TF-IDF content similarity drive recommendations |
| **Full-text Search** | ILIKE search across title and description |
| **PostgreSQL Storage** | Articles, summaries, and user preferences persisted in a relational DB |
| **Streamlit UI** | Four pages: Home feed, Search, Browse by category, Settings |

---

## 🏗️ Project Structure

```
ai-news-aggregator/
│
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example            # Copy to .env and fill in secrets
│
├── config/
│   └── settings.py         # All configuration in one place
│
├── models/
│   └── database.py         # SQLAlchemy ORM models (Article, Summary, User, UserPreference)
│
├── backend/
│   ├── news_fetcher.py     # NewsAPI integration + DB persistence
│   ├── summarizer.py       # T5 summarisation + MarianMT translation
│   └── recommender.py      # TF-IDF content-based recommendations
```

---

## 🚀 Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-news-aggregator.git
cd ai-news-aggregator
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and add your NewsAPI key + PostgreSQL connection string
```

### 5. Set up PostgreSQL

Create the database (the app will auto-create all tables on first run):

```sql
CREATE DATABASE news_aggregator;
```

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 Getting a NewsAPI Key

1. Go to [newsapi.org/register](https://newsapi.org/register)
2. Sign up for a free account (100 requests/day on the free plan)
3. Copy the API key into your `.env` file

---

## 🧠 How It Works

### Summarisation Pipeline

```
Article content / description
        ↓
  T5 tokenizer (truncate to 512 tokens)
        ↓
  T5 model (seq2seq generation)
        ↓
   Summary text (40–150 tokens)
        ↓
  [Optional] MarianMT translation
        ↓
  Stored in summaries table
```

### Recommendation Pipeline

```
User preferences (categories + engagement scores)
        ↓
  Preference-weighted article pool
        ↓
  TF-IDF vectorisation (title + description)
        ↓
  Cosine similarity against seed article
        ↓
  Top-N similar articles returned
```

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Streamlit** – interactive web UI
- **FastAPI** (optional backend API layer, extensible)
- **PostgreSQL + SQLAlchemy** – relational data storage
- **HuggingFace Transformers** – T5, MarianMT
- **scikit-learn** – TF-IDF vectorisation
- **NewsAPI** – live headline source

---

## 📌 Notes

- The T5 model is loaded once at startup and cached in session state — this avoids expensive re-loads on Streamlit reruns.
- Translation models are downloaded lazily (only when a language is first selected) and cached in memory.
- The free NewsAPI plan does not return full article body text, so T5 summarises the `description` field in that case — a paid plan or a scraper would provide richer content.
