"""
semantic_cache.py — FAISS-backed semantic cache (mirrors Salesforce SemanticCache exactly).

Replaces the MD5-hash prefetch in redis_client.py.
One shared instance lives in agent.py and is passed to slow_thinker.

Why FAISS instead of Redis hash keys:
  "Can Dyyota integrate?" and "Can Toyota integrate?" → same doc embedding → HIT
  MD5 hash keys → completely different keys → always MISS

Store flow  (slow thinker after each Pinecone search):
  doc.embedding → normalize_L2 → faiss.IndexFlatIP.add()

Lookup flow (fast talker before every Pinecone call):
  query_embedding → normalize_L2 → faiss.IndexFlatIP.search() → threshold check
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import faiss
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CachedEntry:
    """One cached document chunk with its embedding and TTL."""
    chunks: List[dict]          # the Pinecone chunks returned for this prediction
    embedding: np.ndarray       # L2-normalised embedding stored in FAISS
    relevance_score: float
    created_at: float = field(default_factory=time.time)
    ttl: float = 120.0          # 2 min — matches old REDIS_TTL_PREFETCH
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class SemanticCache:
    """
    In-memory FAISS semantic cache — exact port of Salesforce SemanticCache.

    One instance is shared across the whole agent session.
    Thread-safe for reads; writes are single-threaded (slow thinker is one task).

    Key design choices matching Salesforce:
      - IndexFlatIP  = exact inner product (no approximation needed at this scale)
      - normalize_L2 before every add/search = inner product becomes cosine similarity
      - Keyed by DOCUMENT embedding (not query string or query embedding)
      - TTL checked at read time (no background eviction thread needed)
      - similarity_threshold=0.40 matches Salesforce default for query-to-doc cosine
    """

    def __init__(
        self,
        dimension: int = 1536,          # text-embedding-3-small / ada-002
        max_size: int = 500,
        default_ttl: float = 120.0,     # 2 min per prediction
        similarity_threshold: float = 0.40,  # Salesforce default
    ) -> None:
        self._dimension = dimension
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._similarity_threshold = similarity_threshold

        self._entries: List[CachedEntry] = []
        self._index = faiss.IndexFlatIP(dimension)  # inner product on normalised vecs = cosine

        log.info(
            f"[SemanticCache] init dim={dimension} "
            f"threshold={similarity_threshold} ttl={default_ttl}s"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUT  (slow thinker calls this after each Pinecone search)
    # ─────────────────────────────────────────────────────────────────────────

    def put(
        self,
        doc_embedding: np.ndarray,      # embedding of the DOCUMENT (or prediction query)
        chunks: List[dict],
        relevance_score: float = 1.0,
        ttl: float | None = None,
    ) -> None:
        """
        Store chunks keyed by doc_embedding.
        Matches Salesforce SemanticCache.put() exactly.

        Args:
            doc_embedding: The document's own embedding vector (shape: [dim]).
                           This is what makes "Toyota" and "Dyyota" hit the same entry —
                           both queries are close to the *document* about integrations.
            chunks:        The Pinecone chunks retrieved for this prediction.
            relevance_score: Score from Pinecone (0–1).
            ttl:           Override default TTL in seconds.
        """
        # ── Dedup: if nearly identical embedding already cached, refresh it ──
        if self._index.ntotal > 0:
            query = doc_embedding.reshape(1, -1).astype(np.float32).copy()
            faiss.normalize_L2(query)
            scores, indices = self._index.search(query, 1)
            if scores[0][0] > 0.95 and indices[0][0] != -1:
                idx = int(indices[0][0])
                if idx < len(self._entries):
                    self._entries[idx].chunks = chunks
                    self._entries[idx].relevance_score = relevance_score
                    self._entries[idx].created_at = time.time()
                    log.debug("[SemanticCache] PUT dedup-refreshed existing entry")
                    return

        # ── Evict expired entries ──
        self._evict_expired()

        # ── Evict LRU if at capacity ──
        if len(self._entries) >= self._max_size:
            self._evict_lru()

        # ── Normalise + add to FAISS ──
        embedding = doc_embedding.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(embedding)           # in-place, exact Salesforce pattern
        self._index.add(embedding)

        entry = CachedEntry(
            chunks=chunks,
            embedding=embedding[0],
            relevance_score=relevance_score,
            ttl=ttl or self._default_ttl,
        )
        self._entries.append(entry)
        log.info(
            f"[SemanticCache] PUT {len(chunks)} chunks "
            f"index_size={self._index.ntotal}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GET  (fast talker calls this on every turn before Pinecone)
    # ─────────────────────────────────────────────────────────────────────────

    def get(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float | None = None,
    ) -> Optional[List[dict]]:
        """
        Return chunks for the nearest cached embedding, or None on miss.
        Matches Salesforce SemanticCache.get() exactly.

        Args:
            query_embedding: Live query embedding (shape: [dim]).
            top_k:           Max chunks to return.
            similarity_threshold: Override default threshold.

        Returns:
            List of chunk dicts on HIT, None on MISS.
        """
        threshold = similarity_threshold or self._similarity_threshold

        if self._index.ntotal == 0:
            log.debug("[SemanticCache] MISS (empty index)")
            return None

        query = query_embedding.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(query)                        # same as Salesforce

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)  # cosine similarity scores

        now = time.time()
        best_chunks = None
        best_score = -1.0

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            if entry.is_expired:
                continue
            if score >= threshold and score > best_score:
                best_score = score
                best_chunks = entry.chunks
                entry.access_count += 1

        if best_chunks is not None:
            log.info(
                f"[SemanticCache] HIT  score={best_score:.3f} "
                f"chunks={len(best_chunks)} index_size={self._index.ntotal}"
            )
            return best_chunks

        log.info(
            f"[SemanticCache] MISS best_score={scores[0][0]:.3f} "
            f"< threshold={threshold} index_size={self._index.ntotal}"
        )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # STATS / CLEAR
    # ─────────────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._entries.clear()
        self._index = faiss.IndexFlatIP(self._dimension)
        log.info("[SemanticCache] cleared")

    @property
    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "faiss_total": self._index.ntotal,
            "threshold": self._similarity_threshold,
            "dimension": self._dimension,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────────────────

    def _evict_expired(self) -> int:
        """Remove expired entries and rebuild FAISS index. Returns count removed."""
        live = [i for i, e in enumerate(self._entries) if not e.is_expired]
        removed = len(self._entries) - len(live)
        if removed > 0:
            self._rebuild_index(live)
            log.debug(f"[SemanticCache] evicted {removed} expired entries")
        return removed

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if not self._entries:
            return
        lru_idx = min(range(len(self._entries)),
                      key=lambda i: self._entries[i].access_count)
        live = [i for i in range(len(self._entries)) if i != lru_idx]
        self._rebuild_index(live)
        log.debug("[SemanticCache] evicted LRU entry")

    def _rebuild_index(self, keep_indices: List[int]) -> None:
        """Rebuild FAISS index keeping only the entries at keep_indices."""
        self._entries = [self._entries[i] for i in keep_indices]
        self._index = faiss.IndexFlatIP(self._dimension)
        if self._entries:
            vecs = np.stack([e.embedding for e in self._entries]).astype(np.float32)
            self._index.add(vecs)