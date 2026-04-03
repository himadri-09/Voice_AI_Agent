"""
slow_thinker.py — Background prefetch worker.

Changes:
  1. Guard: skips entirely if query < 3 words (noise turns like "." or "")
     Before: burned 1 LLM + 3 embed + 3 Pinecone calls on garbage input
     After:  logs one line and returns immediately

  2. Fixed log message:
     Before: "15/3 topics prefetched"  ← made no sense (15 chunks / 3 topics)
     After:  "3/3 topics cached (15 chunks total)"  ← clear
"""

import asyncio
import json
import logging
import re
from typing import List, Dict

import numpy as np
from openai import AzureOpenAI
from pinecone import Pinecone

from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME,
    AZURE_EMBEDDING_DEPLOYMENT_NAME,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    SLOW_THINKER_N, TOP_K_CHILDREN,
)
from semantic_cache import SemanticCache

log = logging.getLogger(__name__)

MIN_QUERY_WORDS = 3


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
            log.info(f"[SlowThinker] Predicted: {questions}")
            return questions[:n]
    except Exception as e:
        log.warning(f"[SlowThinker] Prediction failed: {e}")
    return []


async def _embed_search_and_cache(
    oai: AzureOpenAI,
    pc_index,
    cache: SemanticCache,
    question: str,
) -> int:
    """Returns number of chunks cached (0 on failure)."""
    try:
        emb_resp = await asyncio.to_thread(
            oai.embeddings.create,
            model=AZURE_EMBEDDING_DEPLOYMENT_NAME,
            input=question,
        )
        query_vector = emb_resp.data[0].embedding
        query_np     = np.array(query_vector, dtype=np.float32)

        result = await asyncio.to_thread(
            pc_index.query,
            vector=query_vector,
            top_k=TOP_K_CHILDREN,
            include_metadata=True,
            filter={"type": {"$eq": "child"}},
        )

        chunks = []
        for match in result.get("matches", []):
            meta  = match.get("metadata", {})
            score = round(match.get("score", 0), 3)
            chunks.append({
                "content":        meta.get("content", ""),
                "parent_content": "",
                "url":            meta.get("url", ""),
                "title":          meta.get("title", ""),
                "section_title":  meta.get("section_title", ""),
                "heading_path":   meta.get("heading_path", ""),
                "score":          score,
            })

        if not chunks:
            return 0

        cache.put(
            doc_embedding=query_np,
            chunks=chunks,
            relevance_score=chunks[0]["score"],
        )
        log.info(f"[SlowThinker] Cached {len(chunks)} chunks → \"{question[:50]}\"")
        return len(chunks)

    except Exception as e:
        log.warning(f"[SlowThinker] Failed for '{question[:40]}': {e}")
        return 0


async def run_slow_thinker(
    cache: SemanticCache,
    last_query: str,
    last_answer: str,
) -> None:
    """
    Fired with asyncio.create_task() — never awaited by the main loop.

    Guard: returns immediately if query is too short.
    This prevents wasting API calls on noise turns (".", "", one word).
    """
    # ── Guard ──────────────────────────────────────────────────────────────────
    if len(last_query.split()) < MIN_QUERY_WORDS:
        log.info(f"[SlowThinker] Skipped — '{last_query}' is too short")
        return

    log.info(f"[SlowThinker] Starting for: \"{last_query[:55]}\"")
    t0 = __import__("time").time()

    try:
        oai      = _get_oai()
        pc_index = _get_pinecone_index()

        followups = await _predict_followups(oai, last_query, last_answer, SLOW_THINKER_N)
        if not followups:
            log.info("[SlowThinker] No follow-ups — exiting")
            return

        tasks   = [_embed_search_and_cache(oai, pc_index, cache, q) for q in followups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── Fixed log ──────────────────────────────────────────────────────────
        # Before: "15/3 topics prefetched"  (stored = total chunks, not topics)
        # After:  "3/3 topics cached (15 chunks total)"
        chunks_per_topic = [r for r in results if isinstance(r, int) and r > 0]
        topics_ok        = len(chunks_per_topic)
        total_chunks     = sum(chunks_per_topic)
        elapsed          = __import__("time").time() - t0

        log.info(
            f"[SlowThinker] Done — {topics_ok}/{len(followups)} topics cached "
            f"({total_chunks} chunks total) in {elapsed:.2f}s | "
            f"cache size: {cache.size}"
        )

    except Exception as e:
        log.error(f"[SlowThinker] Error: {e}")