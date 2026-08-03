"""
database package initialization.
"""

from database.database import (
    add_to_watchlist,
    get_eod_summary,
    get_recent_signals,
    get_search_history,
    get_watchlist,
    initialise_db,
    is_in_watchlist,
    log_search_event,
    remove_from_watchlist,
    save_signal,
)

__all__ = [
    "add_to_watchlist",
    "get_eod_summary",
    "get_recent_signals",
    "get_search_history",
    "get_watchlist",
    "initialise_db",
    "is_in_watchlist",
    "log_search_event",
    "remove_from_watchlist",
    "save_signal",
]
