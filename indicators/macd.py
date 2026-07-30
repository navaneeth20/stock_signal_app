"""
indicators/macd.py
==================
Moving Average Convergence Divergence (MACD).
"""

from __future__ import annotations

import pandas as pd


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute MACD line, Signal line, and Histogram.

    Args:
        df:     DataFrame with 'Close' column.
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        DataFrame with added columns: MACD, MACD_Signal, MACD_Hist.
    """
    result = df.copy()
    ema_fast = result["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["Close"].ewm(span=slow, adjust=False).mean()
    result["MACD"] = ema_fast - ema_slow
    result["MACD_Signal"] = result["MACD"].ewm(span=signal, adjust=False).mean()
    result["MACD_Hist"] = result["MACD"] - result["MACD_Signal"]
    return result


def macd_signal(df: pd.DataFrame) -> dict:
    """
    Derive MACD-based signal by detecting crossovers and histogram direction.

    Returns:
        dict with signal (1/-1/0), score, reasons.
    """
    required = ["MACD", "MACD_Signal", "MACD_Hist"]
    if not all(c in df.columns for c in required) or len(df) < 2:
        return {"signal": 0, "score": 0, "reasons": ["MACD not available"]}

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    score = 0
    reasons: list[str] = []

    # Bullish crossover: MACD crosses above Signal
    if prev["MACD"] < prev["MACD_Signal"] and curr["MACD"] > curr["MACD_Signal"]:
        score += 2
        reasons.append("MACD bullish crossover (MACD crossed above Signal)")
    # Bearish crossover
    elif prev["MACD"] > prev["MACD_Signal"] and curr["MACD"] < curr["MACD_Signal"]:
        score -= 2
        reasons.append("MACD bearish crossover (MACD crossed below Signal)")
    # No crossover — use position
    elif curr["MACD"] > curr["MACD_Signal"]:
        score += 1
        reasons.append(f"MACD ({curr['MACD']:.4f}) above Signal — Bullish momentum")
    else:
        score -= 1
        reasons.append(f"MACD ({curr['MACD']:.4f}) below Signal — Bearish momentum")

    # Histogram trend
    if curr["MACD_Hist"] > 0 and curr["MACD_Hist"] > prev["MACD_Hist"]:
        score += 1
        reasons.append("Histogram expanding positively — Increasing bullish momentum")
    elif curr["MACD_Hist"] < 0 and curr["MACD_Hist"] < prev["MACD_Hist"]:
        score -= 1
        reasons.append("Histogram contracting negatively — Increasing bearish momentum")

    signal = 1 if score > 0 else (-1 if score < 0 else 0)
    return {"signal": signal, "score": score, "reasons": reasons}
