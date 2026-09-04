"""
data/cache.py
=============
Two-level cache:
  1. In-memory dict with TTL (fast, per-process)
  2. SQLite-backed disk persistence (survives restarts)
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE_DB_PATH = Path(__file__).parent.parent / "database" / "cache.db"
_DEFAULT_TTL = 300  # 5 minutes
_DEFAULT_MAX_MEM_ENTRIES = 256  # L1 bound; each entry is a DataFrame


class DataCache:
    """Thread-safe two-level cache for DataFrame objects."""

    def __init__(
        self,
        ttl: int = _DEFAULT_TTL,
        db_path: Path = _CACHE_DB_PATH,
        max_memory_entries: int = _DEFAULT_MAX_MEM_ENTRIES,
    ) -> None:
        self._ttl = ttl
        # OrderedDict so L1 can evict least-recently-used entries. An unbounded
        # dict of DataFrames grows for the life of the process.
        self._mem: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._max_memory_entries = max_memory_entries
        self._lock = threading.RLock()
        self._db_path = db_path
        self._init_db()
        self.clear_expired()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the SQLite cache table if it doesn't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    expire_at REAL NOT NULL
                )
                """
            )

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), timeout=10)

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.

        Returns None if key is missing or expired.
        Checks memory first, then disk.
        """
        hk = self._hash_key(key)
        now = time.time()

        # L1 – memory
        with self._lock:
            if hk in self._mem:
                val, expire = self._mem[hk]
                if now < expire:
                    self._mem.move_to_end(hk)  # mark as recently used
                    return val
                del self._mem[hk]

        # L2 – disk
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT value, expire_at FROM cache WHERE key = ?", (hk,)
                ).fetchone()
            if row:
                _, expire = row
                if now < expire:
                    val = pickle.loads(row[0])
                    self._remember(hk, val, expire)  # warm L1
                    return val
                # Expired — purge
                self._delete(hk)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cache read error: %s", exc)

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in both memory and disk caches.

        Args:
            key:   Cache key.
            value: Picklable Python object.
            ttl:   Override TTL in seconds.
        """
        hk = self._hash_key(key)
        expire = time.time() + (ttl or self._ttl)
        self._remember(hk, value, expire)
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expire_at) VALUES (?,?,?)",
                    (hk, blob, expire),
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cache write error: %s", exc)

    def _remember(self, hashed_key: str, value: Any, expire: float) -> None:
        """Store in L1, evicting the least-recently-used entry past the bound."""
        with self._lock:
            self._mem[hashed_key] = (value, expire)
            self._mem.move_to_end(hashed_key)
            while len(self._mem) > self._max_memory_entries:
                self._mem.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from both caches."""
        hk = self._hash_key(key)
        with self._lock:
            self._mem.pop(hk, None)
        self._delete(hk)

    def clear_expired(self) -> int:
        """Remove all expired entries from disk cache. Returns count removed."""
        try:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM cache WHERE expire_at < ?", (time.time(),)
                )
                return cur.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cache purge error: %s", exc)
            return 0

    def _delete(self, hashed_key: str) -> None:
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (hashed_key,))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cache delete error: %s", exc)
