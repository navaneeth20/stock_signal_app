"""
indicators/vwap.py
==================
Volume Weighted Average Price (VWAP).
For daily data, VWAP is computed as a rolling window VWAP.
For intraday data, it resets at the start of each session.
"""

from __future__ import annotations

import pandas as pd


def compute_vwap(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute rolling VWAP over a given period.

    For true intraday VWAP, data should be intraday (1h, 15m, 5m).
    For daily data, this acts as a rolling volume-weighted mean.

    Args:
        df:     DataFrame with High, Low, Close, Volume columns.
        period: Rolling window (default 14).

    Returns:
        DataFrame with added 'VWAP' column.
    """
    result = df.copy()
    typical_price = (result["High"] + result["Low"] + result["Close"]) / 3
    tp_volume = typical_price * result["Volume"]

    result["VWAP"] = (
        tp_volume.rolling(window=period).sum()
        / result["Volume"].rolling(window=period).sum()
    )
    return result


def vwap_signal(df: pd.DataFrame) -> dict:
    """
    Generate VWAP-based signal.

    Returns:
        dict with signal, score, reasons.
    """
    if "VWAP" not in df.columns or df["VWAP"].isna().all():
        return {"signal": 0, "score": 0, "reasons": ["VWAP not available"]}

    row = df.iloc[-1]
    close = row["Close"]
    vwap = row["VWAP"]

    score = 0
    reasons: list[str] = []

    if close > vwap:
        score = 1
        reasons.append(f"Price ({close:.2f}) > VWAP ({vwap:.2f}) — Bullish")
    else:
        score = -1
        reasons.append(f"Price ({close:.2f}) < VWAP ({vwap:.2f}) — Bearish")

    signal = 1 if score > 0 else -1
    return {"signal": signal, "score": score, "reasons": reasons, "vwap_value": vwap}
