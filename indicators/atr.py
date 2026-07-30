"""
indicators/atr.py
=================
Average True Range (ATR) — Wilder's smoothing.
"""

from __future__ import annotations

import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ATR using Wilder's smoothing.

    Args:
        df:     DataFrame with High, Low, Close columns.
        period: ATR period (default 14).

    Returns:
        DataFrame with added 'ATR' column.
    """
    result = df.copy()
    high = result["High"]
    low = result["Low"]
    close = result["Close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    result["ATR"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return result
