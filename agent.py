"""
agent.py — Voice AI agent: Deepgram STT → Redis → Pinecone → LLM → Deepgram TTS.

Retrieval priority per turn:
  1. Redis prefetch cache  (slow thinker pre-warmed — instant)
  2. Redis semantic cache  (same/similar query asked before — instant)
  3. Pinecone child search (fresh retrieval — ~300ms)
     → for each child hit, attach parent content for full context

After every turn, slow_thinker runs as a background task to prefetch
likely follow-up chunks into Redis.

Press Ctrl+C to exit.
"""

import asyncio
import io
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import redis
import sounddevice as sd
import soundfile as sf
from deepgram import DeepgramClient
from openai import AzureOpenAI
from pinecone import Pinecone

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
    DEEPGRAM_API_KEY, DEEPGRAM_STT_MODEL, DEEPGRAM_TTS_MODEL,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    TOP_K_CHILDREN,
)
from redis_client import (
    get_redis_client, redis_ping,
    session_add_turn, session_get_history,
    session_save_chunks, session_get_last_chunks,
    semantic_get, semantic_set,
    prefetch_get, prefetch_clear_call,
)
from slow_thinker import run_slow_thinker

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
CHANNELS          = 1
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION  = 2    # seconds of silence to stop recording
MAX_RECORD_SEC    = 30

GREETING = (
    "Hello! I'm your AI assistant. I'm ready to answer your questions about Dyyota "
    "and our services. Just speak after the beep and I'll help you."
)

EXIT_PHRASES = {"exit","quit","bye","goodbye","stop","that's all","end call"}


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

def _dg() -> DeepgramClient:
    return DeepgramClient(api_key=DEEPGRAM_API_KEY)


# ═════════════════════════════════════════════════════════════════════════════
# STT
# ═════════════════════════════════════════════════════════════════════════════

def _beep():
    try:
        t    = np.linspace(0, 0.15, int(SAMPLE_RATE * 0.15), False)
        tone = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        sd.play(tone, SAMPLE_RATE); sd.wait()
    except Exception:
        pass

def record_until_silence() -> Tuple[np.ndarray, float]:
    """Record audio until silence detected. Returns (audio, duration_sec)."""
    print("🎤 Listening...")
    start_time = time.time()
    _beep()
    frames        = []
    silent        = 0
    chunk_size    = int(SAMPLE_RATE * 0.1)
    need_silent   = int(SILENCE_DURATION / 0.1)
    max_chunks    = int(MAX_RECORD_SEC / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=chunk_size) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            silent = silent + 1 if rms < SILENCE_THRESHOLD else 0
            if silent >= need_silent and len(frames) > 10:
                break

    audio = np.concatenate(frames, axis=0)
    duration = len(audio) / SAMPLE_RATE
    print(f"🎤 Recorded {duration:.1f}s")
    return audio, duration

def transcribe(dg: DeepgramClient, audio: np.ndarray) -> Tuple[str, float]:
    """Transcribe audio using Deepgram STT. Returns (transcript, latency_sec)."""
    start_time = time.time()
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    try:
        resp = dg.listen.v1.media.transcribe_file(
            request=buf.read(),
            model=DEEPGRAM_STT_MODEL,
            language="en",
            smart_format=True,
            punctuate=True,
        )
        transcript = resp.results.channels[0].alternatives[0].transcript
        latency = time.time() - start_time
        print(f"📝 You said: {transcript!r}")
        return transcript.strip(), latency
    except Exception as e:
        latency = time.time() - start_time
        log.error(f"STT error: {e}")
        return "", latency


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL  — prefetch → semantic cache → Pinecone children → parent context
# ═════════════════════════════════════════════════════════════════════════════

def _embed_query(oai: AzureOpenAI, query: str) -> List[float]:
    resp = oai.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT_NAME, input=query)
    return resp.data[0].embedding

def _search_pinecone(pc_index, vector: List[float]) -> Tuple[List[Dict], float]:
    """Search child chunks, then fetch parent content. Returns (chunks, latency_sec)."""
    start_time = time.time()
    result = pc_index.query(
        vector=vector,
        top_k=TOP_K_CHILDREN,
        include_metadata=True,
        filter={"type": {"$eq": "child"}},
    )

    chunks = []
    fetched_parents = {}

    for match in result.get("matches", []):
        meta      = match.get("metadata", {})
        parent_id = meta.get("parent_id", "")
        score     = round(match.get("score", 0), 3)

        # Fetch parent content (once per unique parent_id)
        parent_content = ""
        if parent_id:
            if parent_id not in fetched_parents:
                try:
                    fetch_resp = pc_index.fetch(ids=[parent_id])
                    pv         = fetch_resp.get("vectors", {}).get(parent_id)
                    if pv:
                        fetched_parents[parent_id] = pv["metadata"].get("content","")
                    else:
                        fetched_parents[parent_id] = ""
                except Exception:
                    fetched_parents[parent_id] = ""
            parent_content = fetched_parents[parent_id]

        chunks.append({
            "content":        meta.get("content", ""),      # child section
            "parent_content": parent_content,               # full page
            "url":            meta.get("url", ""),
            "title":          meta.get("title", ""),
            "section_title":  meta.get("section_title", ""),
            "heading_path":   meta.get("heading_path", ""),
            "score":          score,
        })

    latency = time.time() - start_time
    print(f"📚 Retrieved {len(chunks)} chunks | top score: {chunks[0]['score'] if chunks else 0} | latency: {latency:.2f}s")
    return chunks, latency

def retrieve(
    r: redis.Redis,
    oai: AzureOpenAI,
    pc_index,
    call_id: str,
    query: str,
) -> Tuple[List[Dict], str, float]:
    """
    Returns (chunks, cache_source, latency_sec) where cache_source is one of:
      "prefetch" | "semantic" | "pinecone"
    """
    start_time = time.time()

    # 1. Prefetch cache (slow thinker pre-warmed)
    prefetched = prefetch_get(r, call_id, query)
    if prefetched:
        latency = time.time() - start_time
        print(f"⚡ Cache: PREFETCH HIT | latency: {latency:.3f}s")
        return prefetched, "prefetch", latency

    # 2. Semantic cache (same query answered recently)
    cached = semantic_get(r, query)
    if cached:
        latency = time.time() - start_time
        print(f"⚡ Cache: SEMANTIC HIT | latency: {latency:.3f}s")
        return cached.get("chunks", []), "semantic", latency

    # 3. Fresh Pinecone retrieval
    print(f"🔍 Cache: MISS → querying Pinecone")
    vector = _embed_query(oai, query)
    chunks, pc_latency = _search_pinecone(pc_index, vector)
    latency = time.time() - start_time
    return chunks, "pinecone", latency


# ═════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_answer(
    oai: AzureOpenAI,
    query: str,
    chunks: List[Dict],
    history: List[Dict],
) -> Tuple[str, float]:
    """Generate answer from LLM. Returns (answer, latency_sec)."""
    start_time = time.time()
    
    if not chunks:
        latency = time.time() - start_time
        return "I'm sorry, I couldn't find relevant information to answer that.", latency

    # Use parent_content when available (full page context), else child content
    context_parts = []
    for i, c in enumerate(chunks):
        body = c.get("parent_content") or c.get("content", "")
        context_parts.append(
            f"[{i+1}] {c['heading_path'] or c['title']}\n{body[:3000]}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a helpful voice assistant for an AI service provider named 'Dyyota'. "
        "Answer the user's question based only on the provided documentation. "
        "Be concise — your answer will be spoken aloud, so keep it to 2-3 sentences max. "
        "Use plain conversational English. No markdown, no bullet points, no code blocks. "
        "If the documentation doesn't contain the answer, say so clearly and briefly."
    )

    messages = [{"role": "system", "content": system}]

    # Include recent conversation history for follow-up context
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
        latency = time.time() - start_time
        print(f"💬 Answer: {answer}")
        return answer, latency
    except Exception as e:
        latency = time.time() - start_time
        log.error(f"Generation error: {e}")
        return "I encountered an error. Please try again.", latency


# ═════════════════════════════════════════════════════════════════════════════
# TTS
# ═════════════════════════════════════════════════════════════════════════════

def speak(dg: DeepgramClient, text: str) -> float:
    """Speak text using Deepgram TTS. Returns latency_sec."""
    start_time = time.time()
    print("🔊 Speaking...")
    try:
        audio_stream = dg.speak.v1.audio.generate(
            text=text,
            model=DEEPGRAM_TTS_MODEL,
        )
        # Collect the audio bytes from the iterator
        audio_bytes = b"".join(audio_stream)
        
        # Write to temporary file
        with open("_tts.mp3", "wb") as f:
            f.write(audio_bytes)
        
        # Play the audio
        data, sr = sf.read("_tts.mp3")
        sd.play(data, sr)
        sd.wait()
        latency = time.time() - start_time
        return latency
    except Exception as e:
        latency = time.time() - start_time
        log.error(f"TTS error: {e}")
        print(f"[TTS fallback] {text}")
        return latency
    finally:
        try:
            os.remove("_tts.mp3")
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_agent():
    print("\n" + "="*55)
    print("🎙️  VOICE RAG AGENT ")
    print("="*55)

    # ── Init clients ──────────────────────────────────────────────────────────
    oai      = _oai()
    pc_index = _pc_index()
    dg       = _dg()
    r        = get_redis_client()

    # ── Health checks ─────────────────────────────────────────────────────────
    if not redis_ping(r):
        print("❌ Redis connection failed. Check REDIS_HOST / REDIS_PASSWORD in .env")
        return

    stats = pc_index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    if total == 0:
        print("⚠️  Pinecone index is empty — run: python ingest.py --url <your-site>")
        return

    print(f"✅ Redis connected")
    print(f"✅ Pinecone ready — {total} vectors")
    print(f"\n🚀 Starting. Say 'exit' or 'goodbye' to end.\n")

    # ── Unique call ID for session isolation ──────────────────────────────────
    call_id = str(uuid.uuid4())[:8]
    print(f"📞 Call ID: {call_id}")

    # ── Greet ─────────────────────────────────────────────────────────────────
    _ = speak(dg, GREETING)

    # ── Conversation loop ─────────────────────────────────────────────────────
    turn = 0
    while True:
        turn += 1
        turn_start = time.time()
        print(f"\n{'─'*40}  Turn {turn}")

        try:
            # 1. Record + transcribe
            audio, record_latency = record_until_silence()
            query, stt_latency = transcribe(dg, audio)
            print(f"  ⏱️  Record: {record_latency:.2f}s | STT: {stt_latency:.2f}s")

            if not query:
                _ = speak(dg, "I didn't catch that. Could you please repeat?")
                continue

            # 2. Exit check
            if any(e in query.lower() for e in EXIT_PHRASES):
                prefetch_clear_call(r, call_id)
                _ = speak(dg, "Goodbye! Have a great day.")
                print("\n👋 Exiting.")
                break

            # 3. Retrieve (prefetch → semantic → Pinecone)
            chunks, cache_source, retrieval_latency = retrieve(r, oai, pc_index, call_id, query)

            # 4. Get conversation history for context
            history = session_get_history(r, call_id)

            # 5. Generate answer
            answer, gen_latency = generate_answer(oai, query, chunks, history)
            print(f"  ⏱️  Retrieval: {retrieval_latency:.2f}s | Generation: {gen_latency:.2f}s")

            # 6. Cache result in semantic cache (only for fresh Pinecone hits)
            if cache_source == "pinecone" and chunks:
                semantic_set(r, query, chunks, answer)

            # 7. Save turn to session history + last chunks
            session_add_turn(r, call_id, "user", query)
            session_add_turn(r, call_id, "assistant", answer)
            session_save_chunks(r, call_id, chunks)

            # 8. Speak answer
            tts_latency = speak(dg, answer)
            print(f"  ⏱️  TTS: {tts_latency:.2f}s")

            # 9. Total turn time (user spoke → agent spoke)
            total_turn_time = time.time() - turn_start
            print(f"  ⏱️  Total Turn Time: {total_turn_time:.2f}s")

            # 10. Fire slow thinker in background (non-blocking)
            asyncio.get_event_loop().run_until_complete(
                asyncio.ensure_future(
                    run_slow_thinker(r, call_id, query, answer)
                )
            )

        except KeyboardInterrupt:
            prefetch_clear_call(r, call_id)
            print("\n\n👋 Interrupted. Exiting.")
            break
        except Exception as e:
            log.error(f"Turn error: {e}")
            try:
                _ = speak(dg, "I encountered an error. Please try again.")
            except Exception:
                pass


if __name__ == "__main__":
    run_agent()