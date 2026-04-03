"""
agent.py — Voice AI agent: STT → SemanticCache (FAISS) → Pinecone → LLM → TTS.

Changes in this version:
  1. Slow thinker guard — skipped if query < 3 words (noise/empty turns)
  2. Clean per-turn log showing the metric that matters:
       Query→Answer = retrieval + generation (what user feels as agent delay)
  3. Suppressed noisy INFO logs from Azure HTTP / SemanticCache internals
     — only WARNINGS and above from third-party libraries are shown
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Tuple

import numpy as np
import sounddevice as sd
from openai import AzureOpenAI
from pinecone import Pinecone
from stt import transcribe
from tts import speak

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    TOP_K_CHILDREN,
)
from redis_client import (
    get_redis_client, redis_ping,
    session_add_turn, session_get_history,
    session_save_chunks,
)
from semantic_cache import SemanticCache
from slow_thinker import run_slow_thinker

# ── Logging: suppress noisy HTTP / cache internals, keep our prints clean ─────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)   # our own logger stays verbose
# Silence Azure SDK and httpx HTTP request lines
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)

# ── Audio settings ─────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
CHANNELS          = 1
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION  = 2
MAX_RECORD_SEC    = 30

EMBEDDING_DIM = 1536   # text-embedding-ada-002 / text-embedding-3-small

GREETING = (
    "Hello! I'm your AI assistant. I'm ready to answer your questions about Dyyota "
    "and our services. Just speak after the beep and I'll help you."
)
EXIT_PHRASES = {"exit", "quit", "bye", "goodbye", "stop", "that's all", "end call"}
MIN_QUERY_WORDS = 3   # below this — skip slow thinker, it's noise


# ═════════════════════════════════════════════════════════════════════════════
# CLIENTS
# ═════════════════════════════════════════════════════════════════════════════

def _oai() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

def _pc_index():
    return Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)


# ═════════════════════════════════════════════════════════════════════════════
# AUDIO
# ═════════════════════════════════════════════════════════════════════════════

def _beep():
    try:
        t    = np.linspace(0, 0.15, int(SAMPLE_RATE * 0.15), False)
        tone = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        sd.play(tone, SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass

def record_until_silence() -> Tuple[np.ndarray, float]:
    print("🎤 Listening...")
    start_time = time.time()
    _beep()
    frames      = []
    silent      = 0
    chunk_size  = int(SAMPLE_RATE * 0.1)
    need_silent = int(SILENCE_DURATION / 0.1)
    max_chunks  = int(MAX_RECORD_SEC / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=chunk_size) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            silent = silent + 1 if rms < SILENCE_THRESHOLD else 0
            if silent >= need_silent and len(frames) > 10:
                break

    audio    = np.concatenate(frames, axis=0)
    duration = len(audio) / SAMPLE_RATE
    return audio, duration


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═════════════════════════════════════════════════════════════════════════════

def _embed_query(oai: AzureOpenAI, query: str) -> np.ndarray:
    resp = oai.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT_NAME,
        input=query,
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _search_pinecone(pc_index, vector: np.ndarray) -> Tuple[List[Dict], float]:
    start = time.time()
    result = pc_index.query(
        vector=vector.tolist(),
        top_k=TOP_K_CHILDREN,
        include_metadata=True,
        filter={"type": {"$eq": "child"}},
    )

    chunks = []
    fetched_parents: Dict[str, str] = {}

    for match in result.get("matches", []):
        meta      = match.get("metadata", {})
        parent_id = meta.get("parent_id", "")
        score     = round(match.get("score", 0), 3)

        parent_content = ""
        if parent_id:
            if parent_id not in fetched_parents:
                try:
                    fetch_resp = pc_index.fetch(ids=[parent_id])
                    pv = fetch_resp.get("vectors", {}).get(parent_id)
                    fetched_parents[parent_id] = pv["metadata"].get("content", "") if pv else ""
                except Exception:
                    fetched_parents[parent_id] = ""
            parent_content = fetched_parents[parent_id]

        chunks.append({
            "content":        meta.get("content", ""),
            "parent_content": parent_content,
            "url":            meta.get("url", ""),
            "title":          meta.get("title", ""),
            "section_title":  meta.get("section_title", ""),
            "heading_path":   meta.get("heading_path", ""),
            "score":          score,
        })

    return chunks, time.time() - start


def retrieve(
    oai: AzureOpenAI,
    pc_index,
    cache: SemanticCache,
    query: str,
) -> Tuple[List[Dict], str, float]:
    start = time.time()
    query_vec = _embed_query(oai, query)

    # 1. FAISS semantic cache
    cached_chunks = cache.get(query_vec)
    if cached_chunks:
        latency = time.time() - start
        return cached_chunks, "cache", latency

    # 2. Pinecone
    chunks, _ = _search_pinecone(pc_index, query_vec)
    if chunks:
        cache.put(doc_embedding=query_vec, chunks=chunks, relevance_score=chunks[0]["score"])

    return chunks, "pinecone", time.time() - start


# ═════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_answer(
    oai: AzureOpenAI,
    query: str,
    chunks: List[Dict],
    history: List[Dict],
) -> Tuple[str, float]:
    start = time.time()

    if not chunks:
        return "I'm sorry, I couldn't find relevant information to answer that.", 0.0

    context_parts = []
    for i, c in enumerate(chunks):
        body = c.get("parent_content") or c.get("content", "")
        context_parts.append(f"[{i+1}] {c['heading_path'] or c['title']}\n{body[:3000]}")
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a helpful voice assistant for Dyyota. "
        "Answer in 2-3 short sentences max — your answer will be spoken aloud. "
        "Use plain conversational English. No markdown, no bullet points, no code blocks. "
        "If the documentation doesn't contain the answer, say so briefly."
    )

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-6:])
    messages.append({
        "role": "user",
        "content": f"Documentation:\n{context}\n\nQuestion: {query}",
    })

    try:
        resp   = oai.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=180,
        )
        answer = resp.choices[0].message.content.strip()
        return answer, time.time() - start
    except Exception as e:
        log.error(f"Generation error: {e}")
        return "I encountered an error. Please try again.", time.time() - start


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

async def run_agent():
    print("\n" + "=" * 55)
    print("🎙️  VOICE RAG AGENT")
    print("=" * 55)

    oai      = _oai()
    pc_index = _pc_index()
    r        = get_redis_client()

    if not redis_ping(r):
        print("❌ Redis connection failed.")
        return

    stats = pc_index.describe_index_stats()
    if stats.get("total_vector_count", 0) == 0:
        print("⚠️  Pinecone index empty — run: python ingest.py --url <your-site>")
        return

    cache = SemanticCache(
        dimension=EMBEDDING_DIM,
        max_size=500,
        default_ttl=120.0,
        similarity_threshold=0.40,
    )

    print(f"✅ Pinecone ready — {stats['total_vector_count']} vectors")
    print(f"✅ SemanticCache ready\n")

    call_id = str(uuid.uuid4())[:8]
    print(f"📞 Call ID: {call_id}\n")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, speak, GREETING)

    turn = 0
    while True:
        turn += 1
        print(f"\n{'─' * 48}  Turn {turn}")

        try:
            # ── 1. Record ─────────────────────────────────────────────────────
            audio, record_sec = await loop.run_in_executor(None, record_until_silence)

            # ── 2. STT ────────────────────────────────────────────────────────
            query, stt_sec = await loop.run_in_executor(None, transcribe, audio)

            # STT done = user's turn is over, agent processing starts now
            agent_start = time.time()

            print(f"🎤  You said: \"{query}\"")
            print(f"    Record {record_sec:.1f}s  |  STT {stt_sec:.2f}s")

            if not query.strip():
                await loop.run_in_executor(None, speak, "I didn't catch that. Could you repeat?")
                continue

            if any(e in query.lower() for e in EXIT_PHRASES):
                cache.clear()
                await loop.run_in_executor(None, speak, "Goodbye! Have a great day.")
                print("\n👋 Exiting.")
                break

            # ── 3. Retrieve ───────────────────────────────────────────────────
            chunks, cache_source, retrieval_sec = await loop.run_in_executor(
                None, retrieve, oai, pc_index, cache, query
            )
            cache_label = f"FAISS hit" if cache_source == "cache" else "Pinecone"

            # ── 4. Generate ───────────────────────────────────────────────────
            history          = session_get_history(r, call_id)
            answer, gen_sec  = await loop.run_in_executor(
                None, generate_answer, oai, query, chunks, history
            )

            # This is the moment the agent has the answer — before TTS API call
            query_to_answer_sec = time.time() - agent_start

            print(f"💬  \"{answer}\"")
            print(f"    Retrieval {retrieval_sec:.2f}s ({cache_label})  |  "
                  f"LLM {gen_sec:.2f}s")
            print(f"    ⚡ Query→Answer: {query_to_answer_sec:.2f}s  "
                  f"← time from your last word to agent having the answer")

            # ── 5. Save session ───────────────────────────────────────────────
            session_add_turn(r, call_id, "user", query)
            session_add_turn(r, call_id, "assistant", answer)
            session_save_chunks(r, call_id, chunks)

            # ── 6. Speak ──────────────────────────────────────────────────────
            print("🔊  Speaking...")
            tts_sec = await loop.run_in_executor(None, speak, answer)
            print(f"    TTS {tts_sec:.2f}s  (audio playback included)")

            # ── 7. Summary ────────────────────────────────────────────────────
            print(f"    Cache entries: {cache.size}")

            # ── 8. Slow thinker — fire only for real queries ──────────────────
            if len(query.split()) >= MIN_QUERY_WORDS:
                asyncio.create_task(run_slow_thinker(cache, query, answer))
            else:
                log.info(f"[SlowThinker] Skipped — query too short: '{query}'")

        except KeyboardInterrupt:
            cache.clear()
            print("\n\n👋 Interrupted.")
            break
        except Exception as e:
            log.error(f"Turn error: {e}", exc_info=True)
            try:
                await loop.run_in_executor(None, speak, "I encountered an error. Please try again.")
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(run_agent())