"""
indicators/ema.py
=================
Exponential Moving Average (EMA) calculations.
"""

from __future__ import annotations

import pandas as pd


def compute_ema(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """
    Add EMA columns to the DataFrame.

    Args:
        df:      DataFrame with a 'Close' column.
        periods: List of EMA periods to compute. Defaults to [20, 50, 200].

    Returns:
        DataFrame with added columns: EMA_<period> for each period.
    """
    if periods is None:
        periods = [20, 50, 200]

    result = df.copy()
    for p in periods:
        result[f"EMA_{p}"] = result["Close"].ewm(span=p, adjust=False).mean()

    return result


def ema_signal(df: pd.DataFrame) -> dict:
    """
    Derive EMA-based trend signal.

    Returns:
        dict with keys: signal (1=bull, -1=bear, 0=neutral), reasons (list[str])
    """
    required = ["Close", "EMA_20", "EMA_50"]
    if not all(c in df.columns for c in required):
        return {"signal": 0, "score": 0, "reasons": ["EMA data not available"]}

    row = df.iloc[-1]
    close = row["Close"]
    ema20 = row["EMA_20"]
    ema50 = row["EMA_50"]

    score = 0
    reasons: list[str] = []

    if ema20 > ema50:
        score += 1
        reasons.append(f"EMA20 ({ema20:.2f}) > EMA50 ({ema50:.2f}) — Bullish")
    else:
        score -= 1
        reasons.append(f"EMA20 ({ema20:.2f}) < EMA50 ({ema50:.2f}) — Bearish")

    if close > ema20:
        score += 1
        reasons.append(f"Price ({close:.2f}) > EMA20 ({ema20:.2f}) — Bullish")
    else:
        score -= 1
        reasons.append(f"Price ({close:.2f}) < EMA20 ({ema20:.2f}) — Bearish")

    # Check EMA200 if available
    if "EMA_200" in df.columns:
        ema200 = row["EMA_200"]
        if close > ema200:
            score += 1
            reasons.append(f"Price > EMA200 ({ema200:.2f}) — Long-term Bullish")
        else:
            score -= 1
            reasons.append(f"Price < EMA200 ({ema200:.2f}) — Long-term Bearish")

    signal = 1 if score > 0 else (-1 if score < 0 else 0)
    return {"signal": signal, "score": score, "reasons": reasons}
