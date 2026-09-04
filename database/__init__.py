"""
database package initialization.
"""

from database.database import (
    MIN_PASSWORD_LENGTH,
    add_to_watchlist,
    authenticate_user,
    count_users,
    get_eod_summary,
    get_recent_signals,
    get_search_history,
    get_user_by_email,
    get_watchlist,
    hash_password,
    initialise_db,
    is_in_watchlist,
    log_search_event,
    register_user,
    remove_from_watchlist,
    save_signal,
    validate_email,
    validate_phone,
    verify_password,
)

__all__ = [
    "MIN_PASSWORD_LENGTH",
    "add_to_watchlist",
    "authenticate_user",
    "count_users",
    "get_eod_summary",
    "get_recent_signals",
    "get_search_history",
    "get_user_by_email",
    "get_watchlist",
    "hash_password",
    "initialise_db",
    "is_in_watchlist",
    "log_search_event",
    "register_user",
    "remove_from_watchlist",
    "save_signal",
    "validate_email",
    "validate_phone",
    "verify_password",
]
