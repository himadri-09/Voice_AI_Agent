# """
# redis_client.py — All Redis operations in one place.

# Key schema:
#   session:{call_id}:history        → JSON list of last 6 turns       TTL=30min
#   session:{call_id}:last_chunks    → JSON list of last retrieved chunks TTL=30min
#   semantic:{md5(query)}            → JSON {chunks, answer}             TTL=5min
#   prefetch:{call_id}:{md5(topic)}  → JSON {chunks}                     TTL=2min
# """

# import hashlib
# import json
# import logging
# from typing import Any, Dict, List, Optional

# import redis

# from config import (
#     REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_SSL,
#     REDIS_TTL_SESSION, REDIS_TTL_SEMANTIC, REDIS_TTL_PREFETCH,
# )

# log = logging.getLogger(__name__)

# # ── Max turns kept in session history ────────────────────────────────────────
# MAX_HISTORY_TURNS = 6


# def get_redis_client() -> redis.Redis:
#     """Return a connected Redis client (Azure Cache for Redis)."""
#     client = redis.Redis(
#         host=REDIS_HOST,
#         port=REDIS_PORT,
#         password=REDIS_PASSWORD,
#         ssl=REDIS_SSL,
#         ssl_cert_reqs=None,          # Azure Redis uses self-signed cert
#         decode_responses=True,
#         socket_connect_timeout=5,
#         socket_timeout=5,
#         retry_on_timeout=True,
#     )
#     return client


# def _query_hash(text: str) -> str:
#     """MD5 hash of lowercased, stripped query — used as cache key."""
#     return hashlib.md5(text.lower().strip().encode()).hexdigest()


# # ═════════════════════════════════════════════════════════════════════════════
# # SESSION CACHE  —  conversation history + last retrieved chunks per call
# # ═════════════════════════════════════════════════════════════════════════════

# def session_add_turn(
#     r: redis.Redis,
#     call_id: str,
#     role: str,        # "user" or "assistant"
#     content: str,
# ) -> None:
#     """Append one turn to the session history list."""
#     key    = f"session:{call_id}:history"
#     turn   = json.dumps({"role": role, "content": content})
#     r.rpush(key, turn)
#     r.ltrim(key, -MAX_HISTORY_TURNS * 2, -1)   # keep last N turns (user+assistant pairs)
#     r.expire(key, REDIS_TTL_SESSION)


# def session_get_history(r: redis.Redis, call_id: str) -> List[Dict]:
#     """Return list of {role, content} dicts for this call."""
#     key  = f"session:{call_id}:history"
#     raw  = r.lrange(key, 0, -1)
#     return [json.loads(t) for t in raw]


# def session_save_chunks(
#     r: redis.Redis,
#     call_id: str,
#     chunks: List[Dict],
# ) -> None:
#     """Save the chunks used in the last answer for this call."""
#     key = f"session:{call_id}:last_chunks"
#     r.set(key, json.dumps(chunks), ex=REDIS_TTL_SESSION)


# def session_get_last_chunks(r: redis.Redis, call_id: str) -> List[Dict]:
#     """Get chunks from the previous turn (may be empty)."""
#     key = f"session:{call_id}:last_chunks"
#     raw = r.get(key)
#     return json.loads(raw) if raw else []


# # ═════════════════════════════════════════════════════════════════════════════
# # SEMANTIC CACHE  —  query → {chunks, answer}
# # ═════════════════════════════════════════════════════════════════════════════

# def semantic_get(r: redis.Redis, query: str) -> Optional[Dict]:
#     """
#     Return cached {chunks, answer} for this query, or None on miss.
#     Uses MD5 of normalized query as key.
#     """
#     key = f"semantic:{_query_hash(query)}"
#     raw = r.get(key)
#     if raw:
#         log.info(f"[Redis] Semantic cache HIT  key={key[:30]}")
#         return json.loads(raw)
#     log.info(f"[Redis] Semantic cache MISS key={key[:30]}")
#     return None


# def semantic_set(
#     r: redis.Redis,
#     query: str,
#     chunks: List[Dict],
#     answer: str,
# ) -> None:
#     """Cache query → {chunks, answer} with semantic TTL."""
#     key     = f"semantic:{_query_hash(query)}"
#     payload = json.dumps({"chunks": chunks, "answer": answer})
#     r.set(key, payload, ex=REDIS_TTL_SEMANTIC)
#     log.info(f"[Redis] Semantic cache SET  key={key[:30]}")


# # ═════════════════════════════════════════════════════════════════════════════
# # PREFETCH CACHE  —  slow thinker pre-warmed chunks
# # ═════════════════════════════════════════════════════════════════════════════

# def prefetch_set(
#     r: redis.Redis,
#     call_id: str,
#     topic: str,
#     chunks: List[Dict],
# ) -> None:
#     """Store pre-fetched chunks for a predicted follow-up topic."""
#     key = f"prefetch:{call_id}:{_query_hash(topic)}"
#     r.set(key, json.dumps({"chunks": chunks}), ex=REDIS_TTL_PREFETCH)
#     log.info(f"[Redis] Prefetch SET  topic={topic[:40]}")


# def prefetch_get(
#     r: redis.Redis,
#     call_id: str,
#     query: str,
# ) -> Optional[List[Dict]]:
#     """
#     Check if any prefetch key for this call matches the current query.
#     Returns chunks list or None.
#     """
#     key = f"prefetch:{call_id}:{_query_hash(query)}"
#     raw = r.get(key)
#     if raw:
#         log.info(f"[Redis] Prefetch HIT  query={query[:40]}")
#         return json.loads(raw).get("chunks", [])
#     return None


# def prefetch_clear_call(r: redis.Redis, call_id: str) -> None:
#     """Delete all prefetch keys for a call (on call end)."""
#     pattern = f"prefetch:{call_id}:*"
#     keys    = r.keys(pattern)
#     if keys:
#         r.delete(*keys)


# # ═════════════════════════════════════════════════════════════════════════════
# # HEALTH CHECK
# # ═════════════════════════════════════════════════════════════════════════════

# def redis_ping(r: redis.Redis) -> bool:
#     try:
#         return r.ping()
#     except Exception as e:
#         log.error(f"[Redis] Ping failed: {e}")
#         return False


"""
redis_client.py — Local in-memory cache (drop-in replacement for Redis).

Swap back to real Azure Redis later by replacing this file only.
All function signatures are identical — agent.py and slow_thinker.py
don't need any changes.

Differences vs Redis version:
  - No network connection needed
  - TTL enforced via timestamps checked on every get
  - Cache lost on process exit (fine for local dev)
  - get_redis_client() returns a _LocalCache object, not redis.Redis
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── TTLs (seconds) ────────────────────────────────────────────────────────────
REDIS_TTL_SESSION  = 1800   # 30 min
REDIS_TTL_SEMANTIC = 300    # 5 min
REDIS_TTL_PREFETCH = 120    # 2 min

MAX_HISTORY_TURNS  = 6


# ═════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORE
# ═════════════════════════════════════════════════════════════════════════════

class _LocalCache:
    """
    Dict-based cache with TTL support.
    Implements the exact methods used by this module so the rest of the
    codebase never knows whether it's talking to Redis or local memory.
    """

    def __init__(self):
        self._store: Dict[str, Dict] = {}

    def _alive(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False
        exp = entry["expires_at"]
        if exp is None:
            return True
        alive = time.time() < exp
        if not alive:
            self._store.pop(key, None)
        return alive

    def get(self, key: str) -> Optional[str]:
        return self._store[key]["value"] if self._alive(key) else None

    def set(self, key: str, value: str, ex: int = None) -> None:
        self._store[key] = {
            "value":      value,
            "expires_at": time.time() + ex if ex else None,
        }

    def rpush(self, key: str, *values) -> None:
        if not self._alive(key):
            self._store[key] = {"value": [], "expires_at": None}
        for v in values:
            self._store[key]["value"].append(v)

    def ltrim(self, key: str, start: int, end: int) -> None:
        if not self._alive(key):
            return
        lst = self._store[key]["value"]
        self._store[key]["value"] = lst[start:] if end == -1 else lst[start:end + 1]

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        if not self._alive(key):
            return []
        lst = self._store[key]["value"]
        return lst[start:] if end == -1 else lst[start:end + 1]

    def expire(self, key: str, seconds: int) -> None:
        if key in self._store:
            self._store[key]["expires_at"] = time.time() + seconds

    def keys(self, pattern: str) -> List[str]:
        import fnmatch
        return [k for k in list(self._store.keys())
                if fnmatch.fnmatch(k, pattern) and self._alive(k)]

    def delete(self, *keys) -> None:
        for k in keys:
            self._store.pop(k, None)

    def ping(self) -> bool:
        return True


# ── Singleton — one cache for the whole process ───────────────────────────────
_cache = _LocalCache()


def get_redis_client() -> _LocalCache:
    """Returns the local in-memory cache (same call as Redis version)."""
    log.info("[Cache] Using local in-memory cache (no Redis)")
    return _cache


def _query_hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# SESSION CACHE
# ═════════════════════════════════════════════════════════════════════════════

def session_add_turn(r: _LocalCache, call_id: str, role: str, content: str) -> None:
    key  = f"session:{call_id}:history"
    turn = json.dumps({"role": role, "content": content})
    r.rpush(key, turn)
    r.ltrim(key, -(MAX_HISTORY_TURNS * 2), -1)
    r.expire(key, REDIS_TTL_SESSION)


def session_get_history(r: _LocalCache, call_id: str) -> List[Dict]:
    key = f"session:{call_id}:history"
    raw = r.lrange(key, 0, -1)
    return [json.loads(t) for t in raw]


def session_save_chunks(r: _LocalCache, call_id: str, chunks: List[Dict]) -> None:
    key = f"session:{call_id}:last_chunks"
    r.set(key, json.dumps(chunks), ex=REDIS_TTL_SESSION)


def session_get_last_chunks(r: _LocalCache, call_id: str) -> List[Dict]:
    key = f"session:{call_id}:last_chunks"
    raw = r.get(key)
    return json.loads(raw) if raw else []


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC CACHE
# ═════════════════════════════════════════════════════════════════════════════

def semantic_get(r: _LocalCache, query: str) -> Optional[Dict]:
    key = f"semantic:{_query_hash(query)}"
    raw = r.get(key)
    if raw:
        log.info(f"[Cache] Semantic HIT  {key[:30]}")
        return json.loads(raw)
    log.info(f"[Cache] Semantic MISS {key[:30]}")
    return None


def semantic_set(r: _LocalCache, query: str, chunks: List[Dict], answer: str) -> None:
    key = f"semantic:{_query_hash(query)}"
    r.set(key, json.dumps({"chunks": chunks, "answer": answer}),
          ex=REDIS_TTL_SEMANTIC)
    log.info(f"[Cache] Semantic SET  {key[:30]}")


# ═════════════════════════════════════════════════════════════════════════════
# PREFETCH CACHE
# ═════════════════════════════════════════════════════════════════════════════

def prefetch_set(r: _LocalCache, call_id: str, topic: str, chunks: List[Dict]) -> None:
    key = f"prefetch:{call_id}:{_query_hash(topic)}"
    r.set(key, json.dumps({"chunks": chunks}), ex=REDIS_TTL_PREFETCH)
    log.info(f"[Cache] Prefetch SET  {topic[:40]}")


def prefetch_get(r: _LocalCache, call_id: str, query: str) -> Optional[List[Dict]]:
    key = f"prefetch:{call_id}:{_query_hash(query)}"
    raw = r.get(key)
    if raw:
        log.info(f"[Cache] Prefetch HIT  {query[:40]}")
        return json.loads(raw).get("chunks", [])
    return None


def prefetch_clear_call(r: _LocalCache, call_id: str) -> None:
    keys = r.keys(f"prefetch:{call_id}:*")
    if keys:
        r.delete(*keys)


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

def redis_ping(r: _LocalCache) -> bool:
    return r.ping()