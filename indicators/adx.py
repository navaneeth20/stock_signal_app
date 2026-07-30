"""
indicators/adx.py
=================
Average Directional Index (ADX) + DI+/DI- using Wilder's smoothing.
"""

from __future__ import annotations

import pandas as pd


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX, DI+, and DI-.

    Args:
        df:     DataFrame with High, Low, Close columns.
        period: ADX smoothing period (default 14).

    Returns:
        DataFrame with added columns: ADX, DI_Plus, DI_Minus.
    """
    result = df.copy()
    high = result["High"]
    low = result["Low"]
    close = result["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1 / period
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    pos_dm_smooth = pos_dm.ewm(alpha=alpha, adjust=False).mean()
    neg_dm_smooth = neg_dm.ewm(alpha=alpha, adjust=False).mean()

    result["DI_Plus"] = 100 * pos_dm_smooth / atr_smooth
    result["DI_Minus"] = 100 * neg_dm_smooth / atr_smooth

    dx = 100 * (result["DI_Plus"] - result["DI_Minus"]).abs() / (
        result["DI_Plus"] + result["DI_Minus"]
    )
    result["ADX"] = dx.ewm(alpha=alpha, adjust=False).mean()
    return result


def adx_signal(df: pd.DataFrame, threshold: float = 25.0) -> dict:
    """
    Generate ADX-based signal.

    Args:
        df:        DataFrame with ADX, DI_Plus, DI_Minus columns.
        threshold: ADX strength threshold (default 25 = trending).

    Returns:
        dict with signal, score, reasons.
    """
    required = ["ADX", "DI_Plus", "DI_Minus"]
    if not all(c in df.columns for c in required):
        return {"signal": 0, "score": 0, "reasons": ["ADX not available"]}

    row = df.iloc[-1]
    adx = row["ADX"]
    di_plus = row["DI_Plus"]
    di_minus = row["DI_Minus"]

    score = 0
    reasons: list[str] = []

    if adx >= threshold:
        reasons.append(f"ADX {adx:.1f} — Strong trending market")
        if di_plus > di_minus:
            score = 1
            reasons.append(f"DI+ ({di_plus:.1f}) > DI- ({di_minus:.1f}) — Bullish trend")
        else:
            score = -1
            reasons.append(f"DI- ({di_minus:.1f}) > DI+ ({di_plus:.1f}) — Bearish trend")
    else:
        reasons.append(f"ADX {adx:.1f} — Weak/ranging market (below {threshold})")

    signal = 1 if score > 0 else (-1 if score < 0 else 0)
    return {"signal": signal, "score": score, "reasons": reasons, "adx_value": adx}
