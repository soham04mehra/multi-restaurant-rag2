# 🍽️ Restaurant AI Chatbot

An AI-powered multi-tenant restaurant chatbot backend that lets customers explore menus, ask questions, and get dish recommendations through natural conversation.

Built on a high-performance **Asynchronous RAG (Retrieval-Augmented Generation)** pipeline, the chatbot guarantees real-time, low-latency, and grounded responses. It leverages **Redis** for sub-millisecond answer caching and persistent conversational session memory, and uses **Supabase (pgvector)** for vector similarity search.

Designed for production scalability, it uses **FastAPI** with non-blocking async execution throughout the entire data flow.

---

## 🏗️ Architecture

### Data Workflow

```text
User Query
    ↓
Redis Cache Check
    ↓ (cache miss)
Retriever (Supabase Vector Search)
    ↓
LLM (Groq / Gemini)
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
Redis Async Calls (Session/Cache)
    ↓
Supabase Query (Vector Similarity RPC)
    ↓
LLM Response (ainvoke)
```

---

## ⚙️ Tech Stack

| Layer | Technology | Why / Purpose |
|---|---|---|
| **API** | FastAPI | High-performance, async-native web framework with auto OpenAPI docs |
| **ASGI Server** | Uvicorn | Lightweight, production-grade asynchronous server |
| **LLM** | Gemini 2.5 Flash / Llama 3.3 70B (Groq) | Primary LLM via Groq with Gemini as robust fallback |
| **Cache & Memory** | Redis | Sub-millisecond session memory & LLM response cache |
| **Asynchronous Client** | `redis.asyncio` | Native Python asyncio client for non-blocking Redis access |
| **Concurrency** | AsyncIO | Core library for asynchronous I/O execution loop |
| **Vector DB** | Supabase pgvector | Managed Postgres with vector search capabilities |
| **Embeddings** | FastEmbed `BAAI/bge-small-en-v1.5` | 384-dimension vector embedding generation |
| **Orchestration** | LangChain | Prompt templates and message history structures |
| **Validation** | Pydantic | JSON schema validation and serialization |
| **Environment** | python-dotenv | Environment variable configuration |

---

## 🚀 Core Features

- 🔍 **Asynchronous Semantic Search** — Non-blocking search using local vector embeddings and database queries.
- 🥗 **Strict Pre-Filtering** — Veg/non-veg, restaurant tenant boundaries, and pricing filters are enforced at the database layer before LLM processing.
- 🚫 **Allergen Safety Checks** — Simple intent-driven keyword matching prevents dangerous food recommendations for allergic customers.
- 🧠 **Redis Session Memory** — Stores multi-turn chat logs in Redis so chatbot memory survives server restarts.
- ⚡ **Redis Answer Caching** — Speeds up repeated queries, reducing latency to single digit milliseconds and protecting LLM/database resources.
- 💬 **Hinglish & Multi-tenant Support** — Isolated data partition per restaurant ID, with multi-lingual conversational understanding.

---

## 💾 Redis Architecture

Redis serves as the backend's state and performance layer, handling both answer caching and session memory.

```text
               ┌────────────────────────┐
               │    FastAPI Backend     │
               └───────────┬────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
 ┌──────────────────────┐    ┌──────────────────────┐
 │    Session Memory    │    │     Answer Cache     │
 │  "session:{sess_id}" │    │ "cache:{rest}:{q}"   │
 └──────────────────────┘    └──────────────────────┘
```

### 1. Session Memory Implementation
To provide natural multi-turn conversations, the chatbot keeps track of message histories.
* **Storage:** Chat histories are serialized to JSON list format and stored in Redis under the key structure `session:{session_id}`.
* **Retrieval (`get_session_history`):** The chat handler performs an async retrieval (`redis_client.get`) at the start of every request, reconstructing a LangChain `ChatMessageHistory` object.
* **Saving (`save_session_history`):** When the message cycle finishes, the updated history is serialized back to JSON and stored asynchronously.

### 2. Cache Implementation
To bypass expensive vector searches and LLM generation for common/repetitive queries, the chatbot caches responses.
* **Caching (`get_cached_answer` / `save_answer_to_cache`):** Before performing any vector database retrieval or LLM inference, the handler queries Redis for the key `cache:{restaurant_id}:{search_query}`.
* **Cache Hit:** If found, the cached JSON payload (containing both the LLM's text response and the list of raw dishes) is returned instantly.
* **Cache Miss:** If not found, the query undergoes standard RAG processing, and the results are stored in the cache asynchronously at the end of the request.

### 3. TTL (Time-To-Live) Configuration
* **Value:** `3600` seconds (1 hour).
* **Resetting:** For session memory, the TTL is reset to 1 hour on every new request using `redis_client.expire()`, ensuring active sessions stay alive while inactive sessions expire automatically to save memory.
* **Cache Expiry:** Cached answers also expire after 1 hour, ensuring that menu changes or updates propagate to users relatively quickly.

### 4. Key Structure & Naming Conventions
* **Session Memory Key:** `session:{session_id}`
* **Answer Cache Key:** `cache:{restaurant_id}:{search_query}`

### 5. Fallback Behavior
If the Redis server goes offline:
* The constructor initialization logs a warning: `Warning: Could not connect to Redis.`
* Runtime database errors are caught gracefully within `try-except` blocks.
* If Redis is unavailable, the chatbot automatically falls back to an in-memory dictionary (`store`) to maintain session memory, ensuring the service remains functional (though memory resets on server restart).

### 6. Latency & Scalability Benefits
* **Sub-millisecond Latency:** Bypasses LLM token generation (which takes 1-3 seconds) and vector search, returning cached results in <5ms.
* **Scale Protection:** Shields Supabase and Groq/Gemini APIs from load spikes, reducing API costs and rate limiting issues.

---

## ⚡ Asynchronous Architecture

The entire RAG pipeline is built using Python's `asyncio` loop to support high concurrent workloads and prevent I/O blocking.

### 1. Why Asynchronous Programming?
A typical RAG pipeline is highly I/O bound. Each request must wait for:
1. Redis cache lookup.
2. Vector database similarity search (Supabase).
3. LLM API execution.
Using synchronous code would block the server's thread pool during these waits, making it impossible to handle multiple users concurrently. Asynchronous programming releases the CPU thread to process other incoming requests while waiting for these network calls.

### 2. Async Modules & Call Graphs
* **`api.py` (FastAPI Endpoints):**
  Defines async routes: `@app.get("/menu/{restaurant_id}")` and `@app.post("/chat")`. FastAPI runs these on the asyncio event loop directly.
* **`chatbot.py` (Core Pipeline):**
  - `get_answer(...)`: Main orchestrator orchestrating async helper tasks.
  - `search_menu(...)`: Queries Supabase asynchronously.
  - `get_session_history(...)` / `save_session_history(...)`: Performs async calls to Redis.
  - `llm.ainvoke(...)`: Non-blocking async invocations to LLM API (Groq/Gemini).
* **`database.py` (Supabase Async Wrapper):**
  Initializes an `AsyncClient` via `create_async_client` and executes async CRUD methods (`insert_dish`, `delete_all_dishes`).

### 3. Concurrent Execution & Performance
* **FastAPI Async Reloader:** Supports hot reloading and concurrent connections using the Uvicorn ASGI server.
* **Async Ingestion Pipeline (`ingest.py`):** Utilizes `asyncio.run()` to batch-upload menus.
* **Asynchronous Sleep (`test_chatbot.py`):** Uses `await asyncio.sleep(35)` during testing to gracefully handle API rate limits without blocking the thread.

---

## 🗂️ Project Structure

```text
├── api.py              # FastAPI endpoints (/chat, /menu, /health)
├── chatbot.py          # Core RAG pipeline (caching, history, LLM orchestration)
├── embeddings.py       # Embedding model + alias expansion system
├── ingest.py           # Async menu ingestion pipeline
├── database.py         # Supabase connection & async DB queries
├── config.py           # Environment configuration
├── requirements.txt    # Python dependencies
├── .env                # API keys & Redis credentials (not committed)
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Setup Database
Run your SQL table setup in the Supabase SQL Editor. The table should support `pgvector` for 384-dimensional embeddings (matching `BAAI/bge-small-en-v1.5`).

### 4. Ingest Menu Data
```bash
python ingest.py
```

### 5. Start FastAPI Server
```bash
python api.py
```
The server will start on `http://localhost:8000`. You can test endpoints via `http://localhost:8000/docs`.

---

## 📡 API Reference

### `POST /chat`
* **Request:**
  ```json
  {
    "message": "suggest me a spicy veg roll under 300 rupees",
    "restaurant_id": "rest_delhi_01",
    "session_id": "optional-session-id"
  }
  ```
* **Response:**
  ```json
  {
    "answer": "**Harissa Falafel Doner** — ₹329...",
    "dishes": [...],
    "session_id": "abc-123-xyz"
  }
  ```

### `GET /menu/{restaurant_id}`
Returns the full menu context for a specific restaurant tenant.

### `GET /health`
Liveness probe endpoint returning server status.

---

## 🤝 Contributing
Feel free to fork the repository, make changes, and open a Pull Request.

---

## 📄 License
MIT License
