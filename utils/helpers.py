"""
utils/helpers.py
================
General-purpose helper utilities for the Stock Signal App.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def format_inr(value: float, decimals: int = 2) -> str:
    """
    Format a number in Indian currency notation (₹ with lakhs/crores).

    Args:
        value:    Numeric value.
        decimals: Decimal places.

    Returns:
        Formatted string, e.g. '₹2.35 Cr', '₹45.60 L', '₹1,234.56'
    """
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:.{decimals}f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value / 1e5:.{decimals}f} L"
    else:
        return f"₹{value:,.{decimals}f}"


def pct_change(old: float, new: float) -> float:
    """Return percentage change from old to new."""
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100


def date_range_from_period(period: str) -> tuple[datetime, datetime]:
    """
    Convert a period string like '1y', '6mo', '3mo' to (start, end) datetimes.

    Args:
        period: Period string.

    Returns:
        (start_datetime, end_datetime)
    """
    end = datetime.now()
    mapping = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
    }
    delta = mapping.get(period, timedelta(days=365))
    return end - delta, end


def format_volume(vol: float) -> str:
    """Format volume in human-readable form (K/M/B)."""
    if vol >= 1e9:
        return f"{vol / 1e9:.2f}B"
    elif vol >= 1e6:
        return f"{vol / 1e6:.2f}M"
    elif vol >= 1e3:
        return f"{vol / 1e3:.1f}K"
    return str(int(vol))


def color_for_signal(signal: str) -> str:
    """Return a hex color string for a signal label."""
    color_map = {
        "Strong Buy": "#00e676",
        "Buy": "#69f0ae",
        "Hold": "#ffd740",
        "Sell": "#ff6e40",
        "Strong Sell": "#f44336",
    }
    return color_map.get(signal, "#9e9e9e")


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate a string with ellipsis if it exceeds max_len."""
    return text if len(text) <= max_len else text[: max_len - 3] + "…"


def get_52w_position(current: float, low_52w: float, high_52w: float) -> float:
    """Return position in 52-week range as 0–1 fraction."""
    rng = high_52w - low_52w
    if rng <= 0:
        return 0.5
    return max(0.0, min(1.0, (current - low_52w) / rng))
