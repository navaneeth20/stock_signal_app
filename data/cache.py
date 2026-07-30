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
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE_DB_PATH = Path(__file__).parent.parent / "database" / "cache.db"
_DEFAULT_TTL = 300  # 5 minutes


class DataCache:
    """Thread-safe two-level cache for DataFrame objects."""

    def __init__(self, ttl: int = _DEFAULT_TTL, db_path: Path = _CACHE_DB_PATH) -> None:
        self._ttl = ttl
        self._mem: dict[str, tuple[Any, float]] = {}  # key → (value, expire_at)
        self._db_path = db_path
        self._init_db()

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
        if hk in self._mem:
            val, expire = self._mem[hk]
            if now < expire:
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
                    self._mem[hk] = (val, expire)  # warm L1
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
        self._mem[hk] = (value, expire)
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expire_at) VALUES (?,?,?)",
                    (hk, blob, expire),
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cache write error: %s", exc)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from both caches."""
        hk = self._hash_key(key)
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
