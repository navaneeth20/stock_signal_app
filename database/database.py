"""
database/database.py
====================
SQLite database layer for watchlist management and signal history.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DB_PATH

logger = logging.getLogger(__name__)

# ── Password hashing ──────────────────────────────────────────────────────────
# PBKDF2-HMAC-SHA256 from the standard library. Not as good as Argon2 or
# bcrypt, but it needs no extra dependency and is a very long way from the
# previous scheme, which was: no password at all.
_PBKDF2_ITERATIONS = 260_000
_SALT_BYTES = 16

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[0-9][0-9\s\-]{7,17}[0-9]$")

MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    """Return a salted PBKDF2 hash, encoded as 'pbkdf2_sha256$iters$salt$hash'."""
    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: Optional[str]) -> bool:
    """Constant-time check of a password against a stored hash."""
    if not stored:
        return False
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(expected, actual)
    except (ValueError, AttributeError):
        logger.warning("Malformed password hash encountered.")
        return False


def validate_email(email: str) -> bool:
    """True when the string is plausibly an email address."""
    return bool(_EMAIL_RE.match((email or "").strip()))


def validate_phone(phone: str) -> bool:
    """True when the string is plausibly a phone number."""
    return bool(_PHONE_RE.match((phone or "").strip()))


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
                user_id     INTEGER NOT NULL DEFAULT 0,
                symbol      TEXT NOT NULL,
                name        TEXT,
                exchange    TEXT DEFAULT 'NSE',
                added_at    TEXT DEFAULT (datetime('now')),
                notes       TEXT,
                UNIQUE (user_id, symbol)
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
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                phone         TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT,
                created_at    TEXT DEFAULT (datetime('now')),
                last_login    TEXT DEFAULT (datetime('now'))
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

        user_cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if user_cols and "password_hash" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")

        wl_cols = [r["name"] for r in conn.execute("PRAGMA table_info(watchlist)").fetchall()]
        if wl_cols and "user_id" not in wl_cols:
            # Pre-existing rows belonged to nobody in particular; park them on
            # user_id 0 so they remain visible but stop leaking between accounts.
            conn.execute("ALTER TABLE watchlist ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist (user_id, symbol)"
        )

    logger.info("Database initialised at %s", DB_PATH)



# ── Watchlist ──────────────────────────────────────────────────────────────────

def add_to_watchlist(
    symbol: str, name: str = "", exchange: str = "NSE", user_id: int = 0
) -> bool:
    """Add a stock to a user's watchlist. Returns True on success."""
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (user_id, symbol, name, exchange) VALUES (?,?,?,?)",
                (int(user_id), symbol.upper(), name, exchange),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("add_to_watchlist error: %s", exc)
        return False


def remove_from_watchlist(symbol: str, user_id: int = 0) -> bool:
    """Remove a stock from a user's watchlist."""
    try:
        with _get_conn() as conn:
            conn.execute(
                "DELETE FROM watchlist WHERE symbol = ? AND user_id = ?",
                (symbol.upper(), int(user_id)),
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("remove_from_watchlist error: %s", exc)
        return False


def get_watchlist(user_id: int = 0) -> list[dict]:
    """Return one user's watchlist entries as a list of dicts."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol, name, exchange, added_at FROM watchlist "
                "WHERE user_id = ? ORDER BY added_at DESC",
                (int(user_id),),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.error("get_watchlist error: %s", exc)
        return []


def is_in_watchlist(symbol: str, user_id: int = 0) -> bool:
    """Check if a symbol is in a user's watchlist."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE symbol = ? AND user_id = ?",
            (symbol.upper(), int(user_id)),
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

def _public_user(row: sqlite3.Row | dict) -> dict:
    """Strip the password hash before a record leaves this module."""
    data = dict(row)
    data.pop("password_hash", None)
    return data


def get_user_by_email(email: str) -> Optional[dict]:
    """
    Retrieve a user record by email, without the password hash.

    Internal only. Never expose this to an unauthenticated caller: it is a
    user-enumeration primitive.
    """
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)
            ).fetchone()
            return _public_user(row) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.error("get_user_by_email error: %s", exc)
        return None


def register_user(name: str, phone: str, email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """
    Create a new account.

    Returns:
        (ok, message, user_record). ``user_record`` never contains the hash.
    """
    clean_name = (name or "").strip()
    clean_phone = (phone or "").strip()
    clean_email = (email or "").strip().lower()

    if not clean_name:
        return False, "Enter your full name.", None
    if not validate_phone(clean_phone):
        return False, "Enter a valid phone number, e.g. +91 9876543210.", None
    if not validate_email(clean_email):
        return False, "Enter a valid email address.", None
    if len(password or "") < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters.", None

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO users (name, phone, email, password_hash, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clean_name, clean_phone, clean_email, hash_password(password), now_str, now_str),
            )
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists. Sign in instead.", None
    except Exception as exc:  # noqa: BLE001
        logger.error("register_user error: %s", exc)
        return False, "Could not create the account. Try again.", None

    return True, "Account created.", get_user_by_email(clean_email)


def authenticate_user(email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """
    Verify an email/password pair.

    The failure message is deliberately identical for "no such user" and "wrong
    password" so this cannot be used to discover which emails are registered.

    Returns:
        (ok, message, user_record).
    """
    clean_email = (email or "").strip().lower()
    generic_failure = "Email or password is incorrect."

    if not clean_email or not password:
        return False, generic_failure, None

    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (clean_email,)
            ).fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.error("authenticate_user error: %s", exc)
        return False, "Sign-in is temporarily unavailable.", None

    if row is None:
        # Spend comparable time on a dummy hash so timing doesn't leak
        # whether the address exists.
        verify_password(password, hash_password("placeholder"))
        return False, generic_failure, None

    stored_hash = dict(row).get("password_hash")
    if not stored_hash:
        return (
            False,
            "This account predates password sign-in. Please register again with a password.",
            None,
        )

    if not verify_password(password, stored_hash):
        return False, generic_failure, None

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _get_conn() as conn:
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_str, row["id"]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update last_login: %s", exc)

    return True, "Signed in.", _public_user(row)


def count_users() -> int:
    """Number of registered accounts. Safe to show; reveals no identities."""
    try:
        with _get_conn() as conn:
            return int(conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"])
    except Exception as exc:  # noqa: BLE001
        logger.error("count_users error: %s", exc)
        return 0


