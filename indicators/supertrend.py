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

    # Raw (basic) bands
    basic_upper = (hl2 + multiplier * atr).to_numpy(dtype=float)
    basic_lower = (hl2 - multiplier * atr).to_numpy(dtype=float)
    close_arr = result["Close"].to_numpy(dtype=float)

    n = len(result)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)

    if n == 0:
        result["Supertrend"] = supertrend
        result["Supertrend_Direction"] = direction
        return result

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    # Seed the line on the side that matches direction[0] (= 1, uptrend).
    # Seeding with the upper band while calling the trend bullish puts the
    # line above price on bar 0 and breaks the invariant that an uptrend line
    # sits below the highs.
    supertrend[0] = basic_lower[0]

    for i in range(1, n):
        # The final bands are *carried forward*: a band only loosens when the
        # raw band moves in the favourable direction, or when price closes
        # through the previous FINAL band (not the previous raw band).
        if basic_upper[i] < final_upper[i - 1] or close_arr[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or close_arr[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        prev_dir = direction[i - 1]
        close = close_arr[i]

        if prev_dir == 1 and close < final_lower[i]:
            direction[i] = -1
        elif prev_dir == -1 and close > final_upper[i]:
            direction[i] = 1
        else:
            direction[i] = prev_dir

        supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

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
