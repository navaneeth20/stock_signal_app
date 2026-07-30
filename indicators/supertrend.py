"""
indicators/supertrend.py
========================
Supertrend indicator — trend-following overlay based on ATR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.atr import compute_atr


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Compute the Supertrend indicator.

    Args:
        df:          DataFrame with High, Low, Close columns.
        period:      ATR period (default 10).
        multiplier:  ATR multiplier (default 3.0).

    Returns:
        DataFrame with added columns:
          - Supertrend: The Supertrend line value
          - Supertrend_Direction: 1 = Uptrend, -1 = Downtrend
    """
    result = compute_atr(df, period=period).copy()
    hl2 = (result["High"] + result["Low"]) / 2
    atr = result["ATR"]

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = [np.nan] * len(result)
    direction = [1] * len(result)

    for i in range(1, len(result)):
        close = result["Close"].iloc[i]

        # Upper band
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or result["Close"].iloc[i - 1] > upper_band.iloc[i - 1]:
            ub = upper_band.iloc[i]
        else:
            ub = upper_band.iloc[i - 1]

        # Lower band
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or result["Close"].iloc[i - 1] < lower_band.iloc[i - 1]:
            lb = lower_band.iloc[i]
        else:
            lb = lower_band.iloc[i - 1]

        if not np.isnan(supertrend[i - 1]):
            prev_st = supertrend[i - 1]
            prev_dir = direction[i - 1]
        else:
            prev_st = ub
            prev_dir = 1

        if prev_dir == -1 and close > ub:
            direction[i] = 1
            supertrend[i] = lb
        elif prev_dir == 1 and close < lb:
            direction[i] = -1
            supertrend[i] = ub
        else:
            direction[i] = prev_dir
            supertrend[i] = lb if prev_dir == 1 else ub

    result["Supertrend"] = supertrend
    result["Supertrend_Direction"] = direction
    return result


def supertrend_signal(df: pd.DataFrame) -> dict:
    """
    Generate Supertrend-based signal.

    Returns:
        dict with signal (1=Buy, -1=Sell), score, reasons.
    """
    if "Supertrend_Direction" not in df.columns:
        return {"signal": 0, "score": 0, "reasons": ["Supertrend not available"]}

    curr = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else curr

    direction = int(curr["Supertrend_Direction"])
    prev_direction = int(prev["Supertrend_Direction"])
    close = curr["Close"]
    st_val = curr["Supertrend"]

    score = 0
    reasons: list[str] = []

    if direction == 1:
        score = 2
        if prev_direction == -1:
            reasons.append(f"Supertrend JUST flipped to BUY at ₹{close:.2f} (strong signal!)")
        else:
            reasons.append(f"Supertrend BUY — price above ST support ({st_val:.2f})")
    else:
        score = -2
        if prev_direction == 1:
            reasons.append(f"Supertrend JUST flipped to SELL at ₹{close:.2f} (strong signal!)")
        else:
            reasons.append(f"Supertrend SELL — price below ST resistance ({st_val:.2f})")

    signal = 1 if score > 0 else -1
    return {"signal": signal, "score": score, "reasons": reasons}
