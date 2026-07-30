"""
indicators/bollinger.py
=======================
Bollinger Bands — 20-period SMA ± N standard deviations.
"""

from __future__ import annotations

import pandas as pd


def compute_bollinger(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Args:
        df:      DataFrame with 'Close' column.
        period:  Moving average period (default 20).
        std_dev: Number of standard deviations (default 2.0).

    Returns:
        DataFrame with added columns: BB_Mid, BB_Upper, BB_Lower, BB_Width, BB_Pct.
    """
    result = df.copy()
    mid = result["Close"].rolling(window=period).mean()
    std = result["Close"].rolling(window=period).std()

    result["BB_Mid"] = mid
    result["BB_Upper"] = mid + std_dev * std
    result["BB_Lower"] = mid - std_dev * std
    result["BB_Width"] = (result["BB_Upper"] - result["BB_Lower"]) / result["BB_Mid"]
    result["BB_Pct"] = (result["Close"] - result["BB_Lower"]) / (
        result["BB_Upper"] - result["BB_Lower"]
    )
    return result


def bollinger_signal(df: pd.DataFrame) -> dict:
    """
    Generate Bollinger Band signal based on price position and band squeeze.

    Returns:
        dict with signal, score, reasons.
    """
    required = ["BB_Upper", "BB_Lower", "BB_Mid", "BB_Pct"]
    if not all(c in df.columns for c in required):
        return {"signal": 0, "score": 0, "reasons": ["Bollinger Bands not available"]}

    row = df.iloc[-1]
    pct = row["BB_Pct"]
    close = row["Close"]
    score = 0
    reasons: list[str] = []

    if pct < 0.05:
        score = -2
        reasons.append(f"Price ({close:.2f}) near/below lower BB — Oversold")
    elif pct < 0.2:
        score = 1
        reasons.append("Price in lower BB zone — potential bounce")
    elif pct > 0.95:
        score = -1
        reasons.append(f"Price ({close:.2f}) near/above upper BB — Overbought")
    elif pct > 0.8:
        score = -1
        reasons.append("Price in upper BB zone — caution")
    else:
        reasons.append(f"Price within BB bands (BB%: {pct:.1%})")

    signal = 1 if score > 0 else (-1 if score < 0 else 0)
    return {"signal": signal, "score": score, "reasons": reasons}
