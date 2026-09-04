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


def rsi_signal(
    df: pd.DataFrame,
    overbought: float = 70,
    oversold: float = 30,
    adx_value: float | None = None,
    trending_threshold: float = 25.0,
) -> dict:
    """
    Derive RSI-based signal.

    This engine is trend-following at its core (EMA + MACD + Supertrend carry
    60% of the weight), so RSI is read as a momentum confirmation by default:
    high RSI is bullish, not a sell.

    The mean-reverting reading (high RSI = overbought = fade it) is only valid
    in a range-bound market, so it is applied only when ADX says the market is
    NOT trending. Pass ``adx_value`` to enable that gate; without it the
    momentum reading is used throughout.

    Scores span the full -2..+2 range so RSI can deliver its configured weight
    in both directions.

    Args:
        df:                 DataFrame with 'RSI' column.
        overbought:         RSI level considered overbought.
        oversold:           RSI level considered oversold.
        adx_value:          Current ADX, used to gate the mean-reversion read.
        trending_threshold: ADX level above which the market counts as trending.

    Returns:
        dict with signal, score, reasons.
    """
    if "RSI" not in df.columns or df["RSI"].isna().all():
        return {"signal": 0, "score": 0, "reasons": ["RSI not available"]}

    rsi = float(df["RSI"].iloc[-1])
    if pd.isna(rsi):
        return {"signal": 0, "score": 0, "reasons": ["RSI not available"]}

    score = 0
    reasons: list[str] = []

    # Ranging market → RSI extremes are reversal signals (mean reversion).
    is_ranging = adx_value is not None and not pd.isna(adx_value) and adx_value < trending_threshold

    if is_ranging and rsi <= oversold:
        score = 2
        reasons.append(
            f"RSI {rsi:.1f} — Oversold in a ranging market (ADX {adx_value:.1f}) — reversal up likely"
        )
    elif is_ranging and rsi >= overbought:
        score = -2
        reasons.append(
            f"RSI {rsi:.1f} — Overbought in a ranging market (ADX {adx_value:.1f}) — fade the move"
        )
    # Trending market (or ADX unknown) → RSI confirms momentum direction.
    elif rsi >= 60:
        score = 2
        reasons.append(f"RSI {rsi:.1f} — Strong bullish momentum")
    elif rsi > 55:
        score = 1
        reasons.append(f"RSI {rsi:.1f} — Bullish territory")
    elif rsi <= 40:
        score = -2
        reasons.append(f"RSI {rsi:.1f} — Strong bearish momentum")
    elif rsi < 45:
        score = -1
        reasons.append(f"RSI {rsi:.1f} — Bearish territory")
    else:
        score = 0
        reasons.append(f"RSI {rsi:.1f} — Neutral zone")

    signal = 1 if score > 0 else (-1 if score < 0 else 0)
    return {"signal": signal, "score": score, "reasons": reasons, "rsi_value": rsi}
