# 🍽️ Restaurant AI Chatbot

An AI-powered restaurant chatbot backend that lets customers explore the menu and get dish recommendations through natural conversation.

Built on a **RAG (Retrieval-Augmented Generation)** pipeline — the AI never guesses. It retrieves real dishes from the menu database and generates answers grounded in actual data.

Designed for:

- 🏪 Restaurants
- 🍔 Cloud kitchens
- 📱 Food ordering apps
- 🤖 AI-powered customer support
- 📋 Smart digital menus

---

# 🏗️ Architecture

```text
Customer Query
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Backend                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Intent & Metadata Extraction                         │
│     • Detects veg/non-veg preference                     │
│     • Extracts price range, allergens, spice level       │
│     • Identifies cuisine preferences                     │
│     • Parses natural language constraints                │
│                                                          │
│  2. Query Expansion                                      │
│     • Expands semantic food terms                        │
│     • Example:                                           │
│       "roll" → shawarma, wrap, doner, pita,              │
│                  frankie, kathi roll                     │
│     • Improves recall during vector search               │
│                                                          │
│  3. Vector Retrieval (Supabase pgvector)                 │
│     • Embedding similarity search                        │
│     • Metadata filtering applied BEFORE retrieval        │
│       - is_veg                                           │
│       - cuisine                                          │
│       - allergens                                        │
│       - min_price / max_price                            │
│       - restaurant_id                                    │
│                                                          │
│  4. Re-ranking Layer                                     │
│     • Boosts highly relevant dishes                      │
│     • Prioritizes spicy dishes when requested            │
│     • Improves final retrieval quality                   │
│     • Ensures best contextual dishes reach the LLM       │
│                                                          │
│  5. LLM Response Generation                              │
│     • Gemini 2.0 Flash generates natural responses       │
│     • Answers grounded ONLY in retrieved menu items      │
│     • Prevents hallucinated dishes or pricing            │
│     • Supports conversational follow-up questions        │
│                                                          │
└──────────────────────────────────────────────────────────┘
      │
      ▼
Natural Language Restaurant Response
```

### Data Workflow

```text
User Query
    ↓
Redis Cache Check
    ↓ (cache miss)
Retriever
    ↓
LLM
    ↓
Redis Cache Store
    ↓
Response
```

### Execution Flow

```text
FastAPI Endpoint
    ↓
Async Processing
    ↓
Redis Async Calls
    ↓
Supabase Query
    ↓
LLM Response
```

---

# 💾 Redis Architecture

Redis is utilized in this project to handle fast, persistent storage and to cache LLM responses, ensuring sub-millisecond lookups.

### 1. Why Redis is Used
- **Session Persistence:** Retains chat history across server restarts.
- **Latency Reduction:** Speeds up repeated queries by skipping the vector search and LLM invocation.

### 2. Session Memory Implementation
* **Retrieval (`get_session_history`):** Loads conversational history from key `session:{session_id}`.
* **Saving (`save_session_history`):** Stores serialized message lists back into Redis.

### 3. Cache Implementation
* **Caching (`get_cached_answer` / `save_answer_to_cache`):** Caches response payloads under `cache:{restaurant_id}:{search_query}`.

### 4. TTL & Key Conventions
* **TTL:** 1 hour (3600 seconds), refreshed for active sessions on every query.
* **Keys:** `session:{session_id}` and `cache:{restaurant_id}:{search_query}`.

### 5. Fallback Behavior
If Redis is down, the code catches the exceptions and falls back to the in-memory `store` dictionary.

---

# ⚡ Asynchronous Architecture

Every external call (database, cache, and LLM) is handled asynchronously to prevent I/O blocking.

### 1. Why Async is Used
RAG applications are I/O bound. Async processing allows the ASGI server (Uvicorn) to release threads while waiting for database or LLM network responses, handling high concurrent user requests.

### 2. Async Modules & Endpoints
* **FastAPI:** `@app.post("/chat")` and `@app.get("/menu/{id}")` are declared as async.
* **Chatbot Logic:** Methods like `get_answer`, `search_menu`, and `get_session_history` are coroutines.
* **Database Wrapper:** `get_supabase()` returns an async connection client.

---

# ⚙️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI | Fast, async, auto Swagger docs |
| ASGI Server | Uvicorn | Lightweight production-ready server |
| LLM | Gemini 2.0 Flash | Strong instruction following, Hinglish support, generous free tier |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` | Better semantic retrieval than MiniLM, CPU friendly, free |
| Vector Database | Supabase pgvector | Managed Postgres with vector search built in |
| Cache & Memory | Redis | Sub-millisecond response caching & persistent session memory |
| Async Redis Client | `redis.asyncio` | Native Python asyncio client for non-blocking Redis operations |
| Concurrency | AsyncIO | Python's standard library for writing single-threaded concurrent code using coroutines |
| Orchestration | LangChain | Chat history, prompt management |
| Validation | Pydantic | Clean request/response schema validation |
| Environment Management | python-dotenv | Secure environment configuration |

---

# ✨ Features

- 🔍 **Semantic search** — finds dishes by meaning, not keyword matching
- 🥗 **Veg / Non-veg filtering** — enforced at database level before LLM sees results
- 💰 **Price filtering** — supports queries like *"under 300 rupees"* or *"above 200"*
- 🚫 **Allergen safety** — detects allergen mentions and excludes unsafe dishes automatically
- 🧠 **Conversation memory** — remembers context within a session for natural follow-up questions
- 💬 **Follow-up aware conversations** — understands references like *"something cheaper"*, *"make it veg"*, or *"spicier than before"*
- 🚫 **Hallucination resistant** — responses are grounded only in retrieved menu data
- 🛡️ **Prompt injection protected** — ignores malicious attempts to override system instructions
- 🏪 **Multi-tenant** — one backend serves multiple restaurants with fully isolated menus
- 🌶️ **Spice-aware** — re-ranks results by spice level when customer asks for spicy food
- 📖 **Alias system** — maps everyday customer language to menu terminology for better retrieval
- 🗣️ **Hinglish friendly** — LLM handles mixed Hindi-English queries naturally
- ⚡ **Low latency retrieval** — optimized embedding search with pgvector
- 📚 **Context-aware responses** — maintains conversational continuity
- 🔄 **Dynamic menu updates** — easily re-ingest menus anytime
- 🧾 **Structured API responses** — frontend-friendly JSON output
- 📊 **RAGAS evaluation support** — retrieval and response quality tested using RAGAS metrics

---

# 🗂️ Project Structure

```text
├── api.py              # FastAPI endpoints (/chat, /menu, /health)
├── chatbot.py          # Core RAG pipeline — retrieval + LLM generation
├── embeddings.py       # Embedding model + alias expansion system
├── ingest.py           # Menu ingestion pipeline
├── database.py         # Supabase connection + queries
├── config.py           # Environment configuration
├── supabase.sql        # pgvector schema + search function
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── README.md           # Project documentation
```

---

# 🔄 RAG Pipeline — Step by Step

## INGESTION FLOW (Runs on Menu Updates)

```text
Menu JSON
   ↓
dish_to_text()
   ↓
alias expansion
   ↓
embedding generation
   ↓
store vectors in Supabase
```

## QUERY FLOW (Runs on Every User Message)

```text
Customer query
   ↓
detect veg intent
   ↓
detect price range
   ↓
detect allergens
   ↓
expand query with aliases
   ↓
embed expanded query
   ↓
vector search in Supabase
   ↓
metadata filtering at DB level
   ↓
spice-aware reranking
   ↓
pass top dishes to Gemini
   ↓
LLM generates grounded answer
   ↓
return answer + dish metadata
```

---

# 🗄️ Database Schema

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

# 🔍 Vector Search Function

Filters applied inside Supabase before returning results:

- `restaurant_id` — isolates each restaurant's menu
- `is_veg` — veg or non-veg only when intent detected
- `max_price / min_price` — price range filtering
- `exclude_allergens` — removes dishes with unsafe allergens
- Ranked by cosine similarity to query embedding
- Returns only top relevant dishes to reduce hallucination risk

---

# 🚀 Getting Started

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## 3. Setup Database

Run the SQL from `supabase.sql` in the Supabase SQL Editor.

This creates:

- `menu_items` table
- `pgvector` extension
- vector similarity search function

## 4. Ingest Menu Data

```bash
python ingest.py
```

## 5. Start FastAPI Server

```bash
python api.py
```

## 6. Open Swagger Docs

```text
http://localhost:8000/docs
```

---

# 📡 API Reference

## `POST /chat`

### Request

```json
{
  "message": "suggest me a spicy veg roll under 300 rupees",
  "restaurant_id": "rest_delhi_01",
  "session_id": "optional-on-first-message"
}
```

### Response

```json
{
  "answer": "Here are some spicy veg options under 300 rupees...",
  "dishes": [...],
  "session_id": "abc-123"
}
```

---

## `GET /menu/{restaurant_id}`

### Response

```json
{
  "restaurant_id": "rest_delhi_01",
  "total_dishes": 10,
  "menu": [...]
}
```

---

## `GET /health`

### Response

```json
{
  "status": "ok",
  "message": "Restaurant chatbot API is running"
}
```

---

# 🧠 Example Queries

```text
suggest me a spicy veg roll under 300
show me chicken shawarma options
what are the best cheesy items
show gluten free dishes
recommend me something spicy and non veg
cheap veg wraps near 200 rupees
```

---

# 🧠 Conversation Memory & Follow-Up Handling

The chatbot supports multi-turn conversations and understands contextual follow-up questions naturally.

Examples:

```text
User: suggest me a spicy veg roll
AI: recommends spicy veg dishes

User: something cheaper
AI: understands the previous context and recommends lower-priced veg rolls

User: make it more cheesy
AI: keeps previous dish context and modifies recommendations
```

The system maintains chat history per session so the LLM can answer contextually instead of treating every message as isolated.

---

# 🧠 How Session Memory Works

```text
First message
→ frontend sends no session_id
→ server generates one and returns it

Next messages
→ frontend sends session_id back
→ server loads chat history for that session
→ LLM sees previous conversation context
→ follow-up questions work naturally
```

---

# 📊 RAG Evaluation & Testing

The project also includes evaluation pipelines using **RAGAS** to measure retrieval and answer quality.

Evaluation helps test:

- Context relevance
- Faithfulness to retrieved data
- Answer correctness
- Retrieval quality
- Hallucination resistance

This ensures the chatbot is not only functional, but also measurable and reliable.

---

# 🛡️ Safety & Reliability

- LLM only answers using retrieved menu context
- Chat history is preserved for contextual follow-up conversations
- Follow-up questions reuse previous conversational state intelligently
- Prompt injection attempts are blocked by strict system instructions
- The chatbot ignores attempts to manipulate or override internal behavior
- Metadata filtering happens before LLM generation
- Allergens can be excluded automatically
- Restaurant data is isolated using restaurant_id filtering
- No hallucinated dishes or fake prices returned
- Responses remain grounded in actual database records

---

# ⚠️ Important Notes

- Never commit your `.env` file — it contains secret keys
- Re-run `ingest.py` every time menu or aliases are updated
- Session history is stored in memory — resets on server restart
- Supabase free tier + Gemini free tier covers early growth comfortably
- Use production-grade session storage like Redis for deployment
- Add authentication before deploying publicly

---

# 📦 Requirements

## Install Dependencies

```bash
pip install fastapi uvicorn supabase langchain langchain-groq langchain-google-genai langchain-community fastembed python-dotenv pydantic redis
```

---

## Core Technologies

| Package | Purpose |
|---|---|
| fastapi | Backend API framework |
| uvicorn | ASGI server for FastAPI |
| supabase | Database and vector storage |
| redis | Caching & session store engine |
| langchain | RAG orchestration framework |
| langchain-groq | Groq LLM integration |
| langchain-google-genai | Gemini integration |
| langchain-community | Community LangChain utilities |
| fastembed | Local embedding generation |
| python-dotenv | Environment variable management |
| pydantic | Data validation and schemas |

---

## Recommended Python Version

```bash
Python 3.10+
```

---

# 📈 Future Improvements

- Redis-based persistent session memory
- Hybrid search (BM25 + Vector Search)
- Admin dashboard for menu management
- Real-time analytics for popular dishes
- Voice-enabled ordering assistant
- WhatsApp / Telegram integration
- Multi-language support
- Docker deployment setup
- Kubernetes deployment support
- Streaming LLM responses

---

# 🤝 Contributing

```bash
# Fork repository
# Create feature branch
# Commit changes
# Open pull request
```

---

# 📄 License

```text
MIT License
```

---

# ⭐ Final Notes

This project demonstrates a production-style RAG architecture tailored specifically for restaurant recommendation systems.

Instead of relying on generic chatbot responses, the system retrieves real menu data, applies metadata-aware filtering, and generates grounded conversational answers.

The result is a safer, more accurate, and scalable restaurant AI assistant.
