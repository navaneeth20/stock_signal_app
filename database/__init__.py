"""
database package initialization.
"""

from database.database import (
    add_to_watchlist,
    create_or_update_user,
    get_all_users,
    get_eod_summary,
    get_recent_signals,
    get_search_history,
    get_user_by_email,
    get_user_by_phone,
    get_watchlist,
    initialise_db,
    is_in_watchlist,
    log_search_event,
    remove_from_watchlist,
    save_signal,
)

__all__ = [
    "add_to_watchlist",
    "create_or_update_user",
    "get_all_users",
    "get_eod_summary",
    "get_recent_signals",
    "get_search_history",
    "get_user_by_email",
    "get_user_by_phone",
    "get_watchlist",
    "initialise_db",
    "is_in_watchlist",
    "log_search_event",
    "remove_from_watchlist",
    "save_signal",
]
