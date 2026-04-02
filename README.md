# Voice RAG Agent

Voice AI agent: Deepgram STT → Redis cache → Pinecone RAG → Azure OpenAI → Deepgram TTS.

## Architecture

```
OFFLINE
  ingest.py → crawl → markdown → parent+child chunks → Pinecone

ONLINE (agent.py per turn)
  Mic → Deepgram STT
      → Redis prefetch cache?   → hit: instant chunks
      → Redis semantic cache?   → hit: instant chunks
      → Pinecone child search   → miss: fresh retrieval
            → fetch parent content for full context
      → Azure OpenAI (answer)
      → Redis: save turn + semantic cache
      → Deepgram TTS → Speaker
      → slow_thinker.py (background task)

BACKGROUND (slow_thinker.py)
  Last query + answer
      → LLM predicts 3 follow-up questions
      → embed + search Pinecone for each
      → store in Redis prefetch cache (TTL 2min)
```

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings from .env |
| `redis_client.py` | All Redis operations (session, semantic, prefetch cache) |
| `ingest.py` | Crawl + parent/child chunk + embed + store in Pinecone |
| `slow_thinker.py` | Background follow-up prefetch worker |
| `agent.py` | Main voice loop |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env  # fill in all values
```

## Step 1 — Ingest

```bash
python ingest.py --url https://docs.yoursite.com
python ingest.py --url https://docs.yoursite.com --max-pages 50
```

Chunking rules:
- Every page → 1 parent chunk (full page)
- Pages ≥ 300 words → also N child chunks (one per H2 section)
- Retrieval searches children for precision, sends parent to LLM for context

## Step 2 — Run agent

```bash
python agent.py
```

Say "exit" or "goodbye" to end the session.

## Redis key schema

```
session:{call_id}:history          TTL 30min  last 6 turns
session:{call_id}:last_chunks      TTL 30min  chunks used in last answer
semantic:{md5(query)}              TTL 5min   query → {chunks, answer}
prefetch:{call_id}:{md5(topic)}    TTL 2min   slow thinker pre-warmed chunks
```