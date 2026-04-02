"""
slow_thinker.py — Background prefetch worker.

After every voice turn:
  1. Takes the last user query + assistant answer
  2. Asks a cheap LLM call: "What are the 3 most likely follow-up questions?"
  3. Embeds each predicted question
  4. Searches Pinecone for matching child chunks
  5. Stores results in Redis prefetch cache (TTL=2min)

On the next turn, agent.py checks prefetch cache first — if the user
asks a predicted follow-up, retrieval is instant (no Pinecone call).

This runs as an asyncio background task — it never blocks the voice response.
"""

import asyncio
import json
import logging
import re
from typing import List, Dict

import redis
from openai import AzureOpenAI
from pinecone import Pinecone

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    SLOW_THINKER_N, TOP_K_CHILDREN,
)
from redis_client import prefetch_set

log = logging.getLogger(__name__)


def _get_oai() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

def _get_pinecone_index():
    return Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)


async def _predict_followups(
    oai: AzureOpenAI,
    query: str,
    answer: str,
    n: int,
) -> List[str]:
    """
    Ask LLM to predict the N most likely follow-up questions.
    Uses a short prompt with low max_tokens to keep this cheap and fast.
    """
    prompt = (
        f"A user asked: \"{query}\"\n"
        f"The assistant answered: \"{answer[:300]}\"\n\n"
        f"List the {n} most likely follow-up questions the user might ask next. "
        f"Return ONLY a JSON array of strings, nothing else.\n"
        f"Example: [\"How do I do X?\", \"What is Y?\", \"Can I Z?\"]"
    )
    try:
        response = await asyncio.to_thread(
            oai.chat.completions.create,
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You predict follow-up questions. Return only a JSON array."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.4,
            max_tokens=150,
        )
        text  = response.choices[0].message.content.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            log.info(f"[SlowThinker] Predicted {len(questions)} follow-ups")
            return questions[:n]
    except Exception as e:
        log.warning(f"[SlowThinker] Follow-up prediction failed: {e}")
    return []


async def _embed_and_search(
    oai: AzureOpenAI,
    pc_index,
    question: str,
) -> List[Dict]:
    """Embed one question and search Pinecone for child chunks."""
    try:
        response = await asyncio.to_thread(
            oai.embeddings.create,
            model=AZURE_EMBEDDING_DEPLOYMENT_NAME,
            input=question,
        )
        vector = response.data[0].embedding

        result = await asyncio.to_thread(
            pc_index.query,
            vector=vector,
            top_k=TOP_K_CHILDREN,
            include_metadata=True,
            filter={"type": {"$eq": "child"}},
        )

        chunks = []
        for match in result.get("matches", []):
            meta = match.get("metadata", {})
            chunks.append({
                "content":       meta.get("content", ""),
                "url":           meta.get("url", ""),
                "title":         meta.get("title", ""),
                "section_title": meta.get("section_title", ""),
                "heading_path":  meta.get("heading_path", ""),
                "parent_id":     meta.get("parent_id", ""),
                "score":         round(match.get("score", 0), 3),
            })
        return chunks

    except Exception as e:
        log.warning(f"[SlowThinker] Search failed for '{question[:40]}': {e}")
        return []


async def run_slow_thinker(
    r: redis.Redis,
    call_id: str,
    last_query: str,
    last_answer: str,
) -> None:
    """
    Main slow thinker coroutine — called as asyncio.create_task() from agent.py.
    Runs entirely in background, never awaited by the main voice loop.
    """
    log.info(f"[SlowThinker] Starting for call={call_id}")

    try:
        oai      = _get_oai()
        pc_index = _get_pinecone_index()

        # 1. Predict follow-up questions
        followups = await _predict_followups(oai, last_query, last_answer, SLOW_THINKER_N)
        if not followups:
            log.info("[SlowThinker] No follow-ups predicted — exiting")
            return

        # 2. For each follow-up: embed + search + cache
        tasks = [_embed_and_search(oai, pc_index, q) for q in followups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        stored = 0
        for question, chunks in zip(followups, results):
            if isinstance(chunks, Exception) or not chunks:
                continue
            await asyncio.to_thread(prefetch_set, r, call_id, question, chunks)
            log.info(f"[SlowThinker] Prefetched {len(chunks)} chunks for: {question[:50]}")
            stored += 1

        log.info(f"[SlowThinker] Done — {stored}/{len(followups)} topics prefetched")

    except Exception as e:
        log.error(f"[SlowThinker] Unexpected error: {e}")