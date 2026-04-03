"""
agent.py — Voice RAG Agent using Deepgram Voice Agent SDK.

Rewritten to use deepgram-sdk exactly as shown in official docs:
  https://developers.deepgram.com/docs/voice-agent

KEY CHANGES FROM PREVIOUS VERSION:
  - Uses DeepgramClient + client.agent.v1.connect() instead of raw websockets
  - Settings sent via connection.send_settings() with typed SDK objects
  - Audio streamed via connection.send_media()
  - Events handled via connection.on(EventType.MESSAGE, handler)
  - OpenAI API key read from OPENAI_API_KEY env var automatically by SDK
  - Sample rate 24000 (matches Deepgram docs)
  - Output container "wav" (matches Deepgram docs)

RAG INTEGRATION:
  FunctionCallRequest → run FAISS + Pinecone → send FunctionCallResponse
  This is the hook that injects your knowledge base into the conversation.

ECHO:
  Handled by Deepgram internally — no need for agent_speaking flags.
  Use headphones for best results on MacBook.

INSTALL:
  pip install deepgram-sdk sounddevice numpy
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Dict, List, Tuple

import numpy as np
import sounddevice as sd
from openai import AzureOpenAI
from pinecone import Pinecone

from deepgram import DeepgramClient, ThinkSettingsV1, SpeakSettingsV1
from deepgram.core.events import EventType
from deepgram.agent.v1.types import (
    AgentV1Settings,
    AgentV1SettingsAgent,
    AgentV1SettingsAudio,
    AgentV1SettingsAudioInput,
    AgentV1SettingsAudioOutput,
    AgentV1SettingsAgentListen,
    AgentV1SettingsAgentThinkOneItem,
    AgentV1SettingsAgentSpeakOneItem,
)
from deepgram.agent.v1.types.agent_v1settings_agent_listen_provider import (
    AgentV1SettingsAgentListenProvider_V1,
)
from deepgram.agent.v1.types.agent_v1settings_agent_speak_one_item_provider import (
    AgentV1SettingsAgentSpeakOneItemProvider_Deepgram,
)

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
    session_save_chunks,
)
from semantic_cache import SemanticCache
from slow_thinker import run_slow_thinker
import asyncio

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 24000   # Deepgram Voice Agent works at 24kHz
CHANNELS         = 1
CHUNK_MS         = 100
CHUNK_FRAMES     = int(SAMPLE_RATE * CHUNK_MS / 1000)

EMBEDDING_DIM    = 1536
MIN_QUERY_WORDS  = 3

SYSTEM_PROMPT = (
    "You are a helpful voice assistant for Dyyota, an AI solutions company. "
    "When a user asks ANYTHING about Dyyota's services, pricing, integrations, "
    "industries, technology, or any factual question, you MUST call the "
    "rag_retrieve function first to get accurate information. "
    "Do NOT answer from memory — always call rag_retrieve for factual questions. "
    "Keep answers to 2-3 short sentences spoken aloud. "
    "Use plain conversational English. No markdown, no bullet points."
)

GREETING = (
    "Hello! I'm your Dyyota AI assistant. "
    "Just start talking and I'll answer your questions."
)

EXIT_PHRASES = {"exit", "quit", "bye", "goodbye", "stop", "end call"}


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
# RAG PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def _embed_query(oai: AzureOpenAI, query: str) -> np.ndarray:
    resp = oai.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT_NAME,
        input=query,
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _search_pinecone(pc_index, vector: np.ndarray) -> List[Dict]:
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
            "heading_path":   meta.get("heading_path", ""),
            "score":          score,
        })
    return chunks


def rag_retrieve(
    oai: AzureOpenAI,
    pc_index,
    cache: SemanticCache,
    query: str,
) -> Tuple[str, str, float, float, float]:
    t0        = time.time()
    query_vec = _embed_query(oai, query)
    embed_sec = time.time() - t0

    t1        = time.time()
    cached    = cache.get(query_vec)
    cache_sec = time.time() - t1

    if cached:
        return _chunks_to_context(cached), "cache", embed_sec, cache_sec, 0.0

    t2           = time.time()
    chunks       = _search_pinecone(pc_index, query_vec)
    pinecone_sec = time.time() - t2

    if chunks:
        cache.put(doc_embedding=query_vec, chunks=chunks, relevance_score=chunks[0]["score"])

    return _chunks_to_context(chunks), "pinecone", embed_sec, cache_sec, pinecone_sec


def _chunks_to_context(chunks: List[Dict]) -> str:
    if not chunks:
        return "No relevant information found in the knowledge base."
    parts = []
    for i, c in enumerate(chunks[:5]):
        body  = c.get("parent_content") or c.get("content", "")
        title = c.get("heading_path") or c.get("title", "")
        parts.append(f"[{i+1}] {title}\n{body[:2000]}")
    return "\n\n---\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT
# ═════════════════════════════════════════════════════════════════════════════

def run_agent():
    print("\n" + "=" * 55)
    print("🎙️  VOICE RAG AGENT  (Deepgram Voice Agent + RAG)")
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
    print(f"✅ SemanticCache (FAISS) ready")

    call_id        = str(uuid.uuid4())[:8]
    turn           = 0
    turn_latencies = []

    # Shared state accessed from the message handler thread
    state = {
        "last_user_query":   "",
        "last_agent_answer": "",
        "turn":              0,
    }

    # ── Build settings using SDK typed objects (exactly as per docs) ───────────
    settings = AgentV1Settings(
        type="Settings",
        audio=AgentV1SettingsAudio(
            input=AgentV1SettingsAudioInput(
                encoding="linear16",
                sample_rate=SAMPLE_RATE,
            ),
            output=AgentV1SettingsAudioOutput(
                encoding="linear16",
                sample_rate=SAMPLE_RATE,
                container="wav",
            ),
        ),
        agent=AgentV1SettingsAgent(
            language="en",
            listen=AgentV1SettingsAgentListen(
                provider=AgentV1SettingsAgentListenProvider_V1(
                    version="v1",
                    type="deepgram",
                    model=DEEPGRAM_STT_MODEL,   # nova-2
                )
            ),
            think=[
                AgentV1SettingsAgentThinkOneItem(
                    provider={
                        "type": "open_ai",
                        "model": "gpt-4o-mini",
                        # SDK reads OPENAI_API_KEY from environment automatically
                    },
                    prompt=SYSTEM_PROMPT,
                    functions=[
                        {
                            "name": "rag_retrieve",
                            "description": (
                                "Retrieve relevant information from the Dyyota knowledge base. "
                                "ALWAYS call this for any question about Dyyota services, "
                                "pricing, integrations, industries, or any factual question."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type":        "string",
                                        "description": "The user's question",
                                    }
                                },
                                "required": ["query"],
                            },
                        }
                    ],
                )
            ],
            speak=[
                AgentV1SettingsAgentSpeakOneItem(
                    provider={
                        "type": "deepgram",
                        "model": DEEPGRAM_TTS_MODEL,   # aura-asteria-en
                    }
                )
            ],
            greeting=GREETING,
        ),
    )

    # ── Speaker output stream ──────────────────────────────────────────────────
    speaker = sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    speaker.start()

    # ── Message handler ────────────────────────────────────────────────────────
    def on_message(message):
        nonlocal turn, turn_latencies

        # Binary audio from Deepgram TTS → play immediately
        if isinstance(message, bytes):
            speaker.write(message)
            return

        msg_type = getattr(message, "type", None)
        if msg_type is None:
            return

        if msg_type == "SettingsApplied":
            print(f"\n✅ Deepgram Voice Agent ready — start talking!\n")
            print(f"📞 Call ID: {call_id}")
            print("🎤  Mic is always on — just start talking\n")

        elif msg_type == "ConversationText":
            role    = getattr(message, "role", "")
            content = getattr(message, "content", "").strip()
            if not content:
                return

            if role == "user":
                state["last_user_query"] = content
                state["turn"] += 1
                print(f"\n{'─'*48}  Turn {state['turn']}")
                print(f"🎤  You said: \"{content}\"")

                if any(e in content.lower() for e in EXIT_PHRASES):
                    print("\n👋 Exiting.")
                    _print_latency_summary(turn_latencies)
                    os._exit(0)

            elif role == "assistant":
                state["last_agent_answer"] = content
                print(f"💬  \"{content}\"")
                session_add_turn(r, call_id, "user", state["last_user_query"])
                session_add_turn(r, call_id, "assistant", content)

                if (state["last_user_query"] and
                        len(state["last_user_query"].split()) >= MIN_QUERY_WORDS):
                    asyncio.run(
                        run_slow_thinker(cache, state["last_user_query"], content)
                    )

        elif msg_type == "FunctionCallRequest":
            fn_name  = getattr(message, "function_name", "")
            fn_id    = getattr(message, "function_call_id", "")
            fn_input = getattr(message, "input", {})

            # input may be a string or dict depending on SDK version
            if isinstance(fn_input, str):
                try:
                    fn_input = json.loads(fn_input)
                except Exception:
                    fn_input = {}

            if fn_name == "rag_retrieve":
                query = fn_input.get("query", state["last_user_query"])
                print(f"🔍  RAG lookup: \"{query}\"")
                t_start = time.time()

                context, cache_source, embed_sec, cache_sec, pinecone_sec = \
                    rag_retrieve(oai, pc_index, cache, query)

                retrieval_sec = embed_sec + cache_sec + pinecone_sec
                cache_label   = "FAISS" if cache_source == "cache" else "Pinecone"
                cache_detail  = (
                    f"FAISS {cache_sec*1000:.0f}ms"
                    if cache_source == "cache"
                    else f"Pinecone {pinecone_sec:.2f}s"
                )
                print(f"    Embed: {embed_sec:.2f}s  |  "
                      f"Cache: {cache_sec*1000:.1f}ms → {cache_detail}  |  "
                      f"Total: {retrieval_sec:.2f}s")

                turn_latencies.append({
                    "turn":          state["turn"],
                    "embed_sec":     round(embed_sec, 3),
                    "cache_sec":     round(cache_sec, 3),
                    "pinecone_sec":  round(pinecone_sec, 2),
                    "cache":         cache_label,
                    "retrieval_sec": round(retrieval_sec, 2),
                })

                # Send result back to Deepgram
                connection.send_function_call_response(
                    function_call_id=fn_id,
                    output=context,
                )
                log.info(f"[RAG] Done in {time.time()-t_start:.2f}s ({cache_label})")

        elif msg_type == "AgentStartedSpeaking":
            print("🔊  Speaking...")

        elif msg_type == "UserStartedSpeaking":
            log.info("[Agent] User started speaking")

        elif msg_type == "Error":
            log.error(f"[Deepgram] Error: {message}")

    def on_error(error):
        log.error(f"[Deepgram] WebSocket error: {error}")

    def on_close(event):
        log.info("[Deepgram] Connection closed")
        speaker.stop()
        speaker.close()
        _print_latency_summary(turn_latencies)

    # ── Connect using SDK ──────────────────────────────────────────────────────
    print(f"✅ Connecting to Deepgram Voice Agent...")

    dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    
    # Use context manager to handle connection lifecycle
    with dg_client.agent.v1.connect() as connection:
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR,   on_error)
        connection.on(EventType.CLOSE,   on_close)

        # Send settings
        connection.send_settings(settings)

        # Start listening for events in background thread
        listener_thread = threading.Thread(
            target=connection.start_listening,
            daemon=True,
        )
        listener_thread.start()

        # Wait for connection to establish
        time.sleep(1)

        # ── Stream mic audio continuously ─────────────────────────────────────────
        print("🎤  Streaming mic... (speak to talk, say 'goodbye' to exit)\n")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_FRAMES,
            ) as mic:
                while True:
                    chunk, _ = mic.read(CHUNK_FRAMES)
                    pcm = chunk.astype(np.int16).tobytes()
                    connection.send_media(pcm)
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted.")
            _print_latency_summary(turn_latencies)
        finally:
            speaker.stop()
            speaker.close()


def _print_latency_summary(turn_latencies: list) -> None:
    if not turn_latencies:
        return
    print(f"\n{'═' * 55}")
    print(f"  SESSION RETRIEVAL SUMMARY  ({len(turn_latencies)} RAG calls)")
    print(f"{'═' * 55}")
    print(f"  {'Turn':<6} {'Embed':>8} {'Cache':>8} {'Source':>10} {'Total':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for t in turn_latencies:
        flag = "⚡" if t["cache"] == "FAISS" else "  "
        print(
            f"  {t['turn']:<6} "
            f"{t['embed_sec']:>7.2f}s "
            f"{t['cache_sec']*1000:>6.0f}ms "
            f"{flag}{t['cache']:>8} "
            f"{t['retrieval_sec']:>7.2f}s"
        )
    hits = sum(1 for t in turn_latencies if t["cache"] == "FAISS")
    avg  = sum(t["retrieval_sec"] for t in turn_latencies) / len(turn_latencies)
    print(f"  {'─'*55}")
    print(f"  Avg retrieval : {avg:.2f}s")
    print(f"  Cache hit rate: {hits}/{len(turn_latencies)} "
          f"({100*hits//len(turn_latencies) if turn_latencies else 0}%)")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    run_agent()