"""
database/database.py
====================
SQLite database layer for watchlist management and signal history.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DB_PATH

logger = logging.getLogger(__name__)


def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_db() -> None:
    """Create all required tables if they don't exist and run schema migrations."""
    with _get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL UNIQUE,
                name        TEXT,
                exchange    TEXT DEFAULT 'NSE',
                added_at    TEXT DEFAULT (datetime('now')),
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS signal_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                signal      TEXT NOT NULL,
                confidence  REAL,
                entry_price REAL,
                stop_loss   REAL,
                take_profit REAL,
                risk_reward REAL,
                interval    TEXT,
                reasons     TEXT,
                mtf_status  TEXT,
                win_prob    REAL,
                source      TEXT DEFAULT 'Signal Terminal',
                generated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS search_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                name        TEXT,
                signal      TEXT,
                confidence  REAL,
                price       REAL,
                source      TEXT DEFAULT 'Search',
                searched_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                phone       TEXT NOT NULL,
                email       TEXT NOT NULL UNIQUE,
                created_at  TEXT DEFAULT (datetime('now')),
                last_login  TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_signal_symbol
                ON signal_history (symbol, generated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_search_history_at
                ON search_history (searched_at DESC);

            CREATE INDEX IF NOT EXISTS idx_users_email
                ON users (email);
            """
        )

        # Automatic schema migration for existing databases
        cursor = conn.execute("PRAGMA table_info(signal_history)")
        existing_cols = [r["name"] for r in cursor.fetchall()]
        if "reasons" not in existing_cols:
            conn.execute("ALTER TABLE signal_history ADD COLUMN reasons TEXT")
        if "mtf_status" not in existing_cols:
            conn.execute("ALTER TABLE signal_history ADD COLUMN mtf_status TEXT")
        if "win_prob" not in existing_cols:
            conn.execute("ALTER TABLE signal_history ADD COLUMN win_prob REAL")
        if "source" not in existing_cols:
            conn.execute("ALTER TABLE signal_history ADD COLUMN source TEXT DEFAULT 'Signal Terminal'")

    logger.info("Database initialised at %s", DB_PATH)



# ── Watchlist ──────────────────────────────────────────────────────────────────

def add_to_watchlist(symbol: str, name: str = "", exchange: str = "NSE") -> bool:
    """Add a stock to the watchlist. Returns True on success."""
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (symbol, name, exchange) VALUES (?,?,?)",
                (symbol.upper(), name, exchange),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("add_to_watchlist error: %s", exc)
        return False


def remove_from_watchlist(symbol: str) -> bool:
    """Remove a stock from the watchlist."""
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("remove_from_watchlist error: %s", exc)
        return False


def get_watchlist() -> list[dict]:
    """Return all watchlist entries as a list of dicts."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol, name, exchange, added_at FROM watchlist ORDER BY added_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.error("get_watchlist error: %s", exc)
        return []


def is_in_watchlist(symbol: str) -> bool:
    """Check if a symbol is in the watchlist."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE symbol = ?", (symbol.upper(),)
        ).fetchone()
    return row is not None


# ── Signal History ─────────────────────────────────────────────────────────────

def save_signal(
    symbol: str,
    signal: str,
    confidence: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    risk_reward: float,
    interval: str = "1d",
    reasons: Optional[list[str] | str] = None,
    mtf_status: Optional[str] = None,
    win_prob: Optional[float] = None,
    source: str = "Signal Terminal",
) -> None:
    """Persist a generated signal to the database."""
    try:
        import json
        reasons_str = json.dumps(reasons) if isinstance(reasons, list) else (reasons or "")
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO signal_history
                (symbol, signal, confidence, entry_price, stop_loss, take_profit, risk_reward, interval, reasons, mtf_status, win_prob, source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol.upper(),
                    signal,
                    confidence,
                    entry_price,
                    stop_loss,
                    take_profit,
                    risk_reward,
                    interval,
                    reasons_str,
                    mtf_status or "",
                    win_prob or 0.0,
                    source,
                ),
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("save_signal error: %s", exc)



def get_recent_signals(symbol: Optional[str] = None, limit: int = 50) -> list[dict]:
    """
    Retrieve recent signals from history.

    Args:
        symbol: Optional filter by symbol.
        limit:  Maximum number of rows.

    Returns:
        List of signal dicts.
    """
    try:
        with _get_conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM signal_history WHERE symbol=? ORDER BY generated_at DESC LIMIT ?",
                    (symbol.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM signal_history ORDER BY generated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.error("get_recent_signals error: %s", exc)
        return []


# ── Search History & EOD Reporting ─────────────────────────────────────────────

def log_search_event(
    symbol: str,
    name: str = "",
    signal: str = "",
    confidence: float = 0.0,
    price: float = 0.0,
    source: str = "Search",
) -> None:
    """Record a user search event into SQLite history."""
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO search_history
                (symbol, name, signal, confidence, price, source)
                VALUES (?,?,?,?,?,?)
                """,
                (symbol.upper(), name, signal, confidence, price, source),
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("log_search_event error: %s", exc)


def get_search_history(limit: int = 100, date_filter: Optional[str] = None) -> list[dict]:
    """Retrieve search history entries from SQLite."""
    try:
        with _get_conn() as conn:
            if date_filter:
                rows = conn.execute(
                    "SELECT * FROM search_history WHERE date(searched_at) = date(?) ORDER BY searched_at DESC LIMIT ?",
                    (date_filter, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM search_history ORDER BY searched_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.error("get_search_history error: %s", exc)
        return []


def get_eod_summary(date_str: Optional[str] = None) -> dict:
    """Generate End-of-Day summary stats for a given date (defaults to today)."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM search_history WHERE date(searched_at) = date(?) ORDER BY searched_at DESC",
                (date_str,),
            ).fetchall()
            data = [dict(r) for r in rows]

            unique_symbols = conn.execute(
                "SELECT COUNT(DISTINCT symbol) as count FROM search_history WHERE date(searched_at) = date(?)",
                (date_str,),
            ).fetchone()["count"]

            top_searched = conn.execute(
                """
                SELECT symbol, name, COUNT(*) as query_count 
                FROM search_history 
                WHERE date(searched_at) = date(?)
                GROUP BY symbol 
                ORDER BY query_count DESC 
                LIMIT 5
                """,
                (date_str,),
            ).fetchall()

            return {
                "date": date_str,
                "total_queries": len(data),
                "unique_stocks": unique_symbols,
                "top_searched": [dict(r) for r in top_searched],
                "all_rows": data,
            }
    except Exception as exc:  # noqa: BLE001
        logger.error("get_eod_summary error: %s", exc)
        return {"date": date_str, "total_queries": 0, "unique_stocks": 0, "top_searched": [], "all_rows": []}


# ── User Management ────────────────────────────────────────────────────────────

def get_user_by_email(email: str) -> Optional[dict]:
    """Retrieve user record by email address."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)
            ).fetchone()
            return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.error("get_user_by_email error: %s", exc)
        return None


def get_user_by_phone(phone: str) -> Optional[dict]:
    """Retrieve user record by phone number."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE phone = ?", (phone.strip(),)
            ).fetchone()
            return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.error("get_user_by_phone error: %s", exc)
        return None


def create_or_update_user(name: str, phone: str, email: str) -> dict:
    """
    Create a new user or update existing user login time and details.

    Args:
        name: Full Name
        phone: Phone Number
        email: Email Address

    Returns:
        User record dictionary.
    """
    clean_email = email.strip().lower()
    clean_name = name.strip()
    clean_phone = phone.strip()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing = get_user_by_email(clean_email)
    try:
        with _get_conn() as conn:
            if existing:
                conn.execute(
                    """
                    UPDATE users 
                    SET name = ?, phone = ?, last_login = ? 
                    WHERE LOWER(email) = LOWER(?)
                    """,
                    (clean_name, clean_phone, now_str, clean_email),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO users (name, phone, email, created_at, last_login)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (clean_name, clean_phone, clean_email, now_str, now_str),
                )
        return get_user_by_email(clean_email) or {
            "name": clean_name,
            "phone": clean_phone,
            "email": clean_email,
            "created_at": now_str,
            "last_login": now_str,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("create_or_update_user error: %s", exc)
        return {
            "name": clean_name,
            "phone": clean_phone,
            "email": clean_email,
            "created_at": now_str,
            "last_login": now_str,
        }


def get_all_users() -> list[dict]:
    """Retrieve all registered users."""
    try:
        with _get_conn() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY last_login DESC").fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.error("get_all_users error: %s", exc)
        return []


