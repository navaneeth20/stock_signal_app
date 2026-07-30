"""
indicators/rsi.py
=================
Relative Strength Index (RSI) — Wilder's smoothing method.
"""

from __future__ import annotations

import pandas as pd


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute RSI using Wilder's smoothing (EWM with alpha=1/period).

    Args:
        df:     DataFrame with 'Close' column.
        period: RSI look-back period (default 14).

    Returns:
        DataFrame with added 'RSI' column.
    """
    result = df.copy()
    delta = result["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result["RSI"] = 100 - (100 / (1 + rs))
    return result


def rsi_signal(df: pd.DataFrame, overbought: float = 70, oversold: float = 30) -> dict:
    """
    Derive RSI-based signal.

    Args:
        df:          DataFrame with 'RSI' column.
        overbought:  RSI level considered overbought.
        oversold:    RSI level considered oversold.

    Returns:
        dict with signal, score, reasons.
    """
    if "RSI" not in df.columns or df["RSI"].isna().all():
        return {"signal": 0, "score": 0, "reasons": ["RSI not available"]}

    rsi = df["RSI"].iloc[-1]
    score = 0
    reasons: list[str] = []

    if rsi < oversold:
        score = -2
        reasons.append(f"RSI {rsi:.1f} — Oversold (possible reversal up)")
    elif rsi < 45:
        score = -1
        reasons.append(f"RSI {rsi:.1f} — Bearish territory")
    elif rsi > overbought:
        score = -1
        reasons.append(f"RSI {rsi:.1f} — Overbought (caution)")
    elif rsi > 55:
        score = 1
        reasons.append(f"RSI {rsi:.1f} — Bullish territory")
    elif 45 <= rsi <= 55:
        score = 0
        reasons.append(f"RSI {rsi:.1f} — Neutral zone")

    signal = 1 if score > 0 else (-1 if score < 0 else 0)
    return {"signal": signal, "score": score, "reasons": reasons, "rsi_value": rsi}
