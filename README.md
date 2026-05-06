# 🍽️ Restaurant AI Chatbot

An AI-powered restaurant chatbot backend that lets customers explore the menu and get dish recommendations through natural conversation.

Built on a **RAG (Retrieval-Augmented Generation)** pipeline — the AI never guesses. It retrieves real dishes from the menu database and generates answers grounded in actual data.

---

## 🏗️ Architecture
Customer Message
↓
┌─────────────────────────────────────────────────────┐
│                    FastAPI Backend                   │
│                                                     │
│  1. Intent Detection                                │
│     └── veg/non-veg, price range, allergens,        │
│         spice level extracted from query            │
│                                                     │
│  2. Query Expansion                                 │
│     └── "roll" → doner, shawarma, wrap, pita,       │
│         frankie, kathi, flatbread...                │
│                                                     │
│  3. Vector Search (Supabase pgvector)               │
│     └── filters applied at DB level:               │
│         is_veg, max_price, min_price, allergens     │
│                                                     │
│  4. Spice Re-ranking                                │
│     └── medium/high spice dishes pushed to top     │
│         when customer asks for spicy food           │
│                                                     │
│  5. LLM Answer Generation (Gemini 2.0 Flash)        │
│     └── generates natural friendly response        │
│         from retrieved dishes only                  │
└─────────────────────────────────────────────────────┘
↓
Customer Gets Answer + Dish Cards

---

## ⚙️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI | Fast, async, auto Swagger docs |
| LLM | Gemini 2.0 Flash | Strong instruction following, Hinglish support, generous free tier |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` | Better semantic retrieval than MiniLM, CPU friendly, free |
| Vector Database | Supabase pgvector | Managed Postgres with vector search built in |
| Orchestration | LangChain | Chat history, prompt management |

---

## ✨ Features

- 🔍 **Semantic search** — finds dishes by meaning, not keyword matching
- 🥗 **Veg / Non-veg filtering** — enforced at database level before LLM sees results
- 💰 **Price filtering** — supports queries like *"under 300 rupees"* or *"above 200"*
- 🚫 **Allergen safety** — detects allergen mentions and excludes unsafe dishes automatically
- 🧠 **Conversation memory** — remembers context within a session for follow-up questions
- 🏪 **Multi-tenant** — one backend serves multiple restaurants with fully isolated menus
- 🌶️ **Spice-aware** — re-ranks results by spice level when customer asks for spicy food
- 📖 **Alias system** — maps everyday customer language to menu terminology for better retrieval
- 🗣️ **Hinglish friendly** — LLM handles mixed Hindi-English queries naturally
- 🛡️ **Prompt injection safe** — system prompt explicitly blocks instruction hijacking

---

## 🗂️ Project Structure
├── api.py          # FastAPI endpoints (/chat, /menu, /health)
├── chatbot.py      # Core RAG pipeline — search, filter, LLM call
├── embeddings.py   # Embedding model + dish alias system
├── ingest.py       # Menu ingestion pipeline
├── database.py     # Supabase client
├── config.py       # Environment config
└── supabase.sql    # Vector search function + table schema

---

## 🔄 RAG Pipeline — Step by Step

INGEST (runs once per menu update)
Menu JSON → dish_to_text() → alias expansion → embed → store in Supabase
QUERY (runs on every customer message)
Customer query
→ detect veg intent        (True / False / None)
→ detect price range       (max_price / min_price / None)
→ detect allergens         (dairy / gluten / nuts / etc)
→ expand query with aliases
→ embed expanded query
→ vector search in Supabase with all filters at DB level
→ spice re-rank
→ pass top dishes to LLM as context
→ LLM generates answer
→ return answer + dish list to frontend


---

## 🗄️ Database Schema

```sql
table: menu_items
├── id            uuid (primary key)
├── restaurant_id text
├── name          text
├── description   text
├── price         float
├── cuisine       text
├── is_veg        boolean
├── spice_level   text
├── allergens     text[]
├── ingredients   text[]
├── content       text       ← enriched text used for embedding
└── embedding     vector(384)
```

---

## 🔍 Vector Search Function

Filters applied inside Supabase before returning results:
- `restaurant_id` — isolates each restaurant's menu
- `is_veg` — veg or non-veg only when intent detected
- `max_price / min_price` — price range filtering
- `exclude_allergens` — removes dishes with unsafe allergens
- Ranked by cosine similarity to query embedding

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Add your keys to .env
GOOGLE_API_KEY=your_gemini_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Run SQL in Supabase SQL Editor to create table and search function
# (see supabase.sql)

# Ingest your menu
python ingest.py

# Start the server
python api.py
```

---

## 📡 API Reference

### `POST /chat`
```json
Request:
{
  "message": "suggest me a spicy veg roll under 300 rupees",
  "restaurant_id": "rest_delhi_01",
  "session_id": "optional-on-first-message"
}

Response:
{
  "answer": "Here are some spicy veg options under 300 rupees...",
  "dishes": [...],
  "session_id": "abc-123"
}
```

### `GET /menu/{restaurant_id}`
```json
Response:
{
  "restaurant_id": "rest_delhi_01",
  "total_dishes": 10,
  "menu": [...]
}
```

### `GET /health`
```json
Response:
{
  "status": "ok",
  "message": "Restaurant chatbot API is running"
}
```

---

## 🧠 How Session Memory Works
First message  → frontend sends no session_id
→ server generates one and returns it
Next messages  → frontend sends session_id back
→ server loads chat history for that session
→ LLM sees full conversation context

---

## ⚠️ Important Notes

- Never commit your `.env` file — it contains secret keys
- Re-run `ingest.py` every time menu or aliases are updated
- Session history is stored in memory — resets on server restart
- Supabase free tier + Gemini free tier covers early growth comfortably

---

# 📦 Requirements

## Install Dependencies

Run the following command to install all required packages:

```bash
pip install fastapi uvicorn supabase langchain langchain-groq langchain-google-genai langchain-community fastembed python-dotenv pydantic
```

## Core Technologies

| Package | Purpose |
|---|---|
| fastapi | Backend API framework |
| uvicorn | ASGI server for FastAPI |
| supabase | Database and vector storage |
| langchain | RAG orchestration framework |
| langchain-groq | Groq LLM integration |
| langchain-google-genai | Gemini integration |
| langchain-community | Community LangChain utilities |
| fastembed | Local embedding generation |
| python-dotenv | Environment variable management |
| pydantic | Data validation and schemas |

## Recommended Python Version

```bash
Python 3.10+
```
