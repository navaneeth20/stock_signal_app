"""
indicators/mtf.py
=================
Multi-Timeframe (MTF) Trend Alignment Engine.
Fetches 1H, 1D, and 1W data to determine multi-timeframe trend confluence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from data.fetch_data import fetch_ohlcv
from indicators.ema import compute_ema
from indicators.supertrend import compute_supertrend

logger = logging.getLogger(__name__)


@dataclass
class TimeframeTrend:
    timeframe: str          # "1W", "1D", "1H"
    trend: str              # "Bullish", "Bearish", "Neutral"
    ema_aligned: bool
    supertrend_direction: int  # 1 (Bullish), -1 (Bearish)
    close_price: float


@dataclass
class MTFResult:
    symbol: str
    trends: dict[str, TimeframeTrend]  # "1W", "1D", "1H" -> TimeframeTrend
    alignment_status: str               # "Full Bullish", "Full Bearish", "Partial Bullish", "Partial Bearish", "Divergent"
    alignment_score: float              # -1.0 (Full Bearish) to +1.0 (Full Bullish)
    confidence_modifier: float          # -15.0 to +15.0 %
    description: str


def _analyze_single_tf(df: pd.DataFrame, tf_label: str) -> TimeframeTrend:
    """Analyze trend for a single timeframe DataFrame."""
    if len(df) < 20:
        return TimeframeTrend(tf_label, "Neutral", False, 0, 0.0)

    df = compute_ema(df, periods=[20, 50])
    df = compute_supertrend(df, period=10, multiplier=3.0)

    last = df.iloc[-1]
    close = float(last["Close"])
    ema20 = float(last.get("EMA_20", close))
    ema50 = float(last.get("EMA_50", close))
    st_dir = int(last.get("Supertrend_Direction", 0))

    ema_bullish = ema20 > ema50
    st_bullish = st_dir == 1

    if ema_bullish and st_bullish:
        trend = "Bullish"
    elif not ema_bullish and st_dir == -1:
        trend = "Bearish"
    else:
        trend = "Neutral"

    return TimeframeTrend(
        timeframe=tf_label,
        trend=trend,
        ema_aligned=ema_bullish,
        supertrend_direction=st_dir,
        close_price=close,
    )


def compute_mtf_alignment(symbol: str) -> MTFResult:
    """
    Fetch and analyze Weekly (1W), Daily (1D), and Hourly (1H) trends for a symbol.

    Args:
        symbol: Ticker symbol (e.g. RELIANCE.NS)

    Returns:
        MTFResult object containing trend analysis and alignment score.
    """
    trends = {}

    # 1. Weekly Trend (Macro)
    try:
        df_1w = fetch_ohlcv(symbol, period="2y", interval="1wk")
        trends["1W"] = _analyze_single_tf(df_1w, "1W (Macro)")
    except Exception as exc:
        logger.warning("Failed to fetch 1W data for %s: %s", symbol, exc)
        trends["1W"] = TimeframeTrend("1W (Macro)", "Neutral", False, 0, 0.0)

    # 2. Daily Trend (Setup)
    try:
        df_1d = fetch_ohlcv(symbol, period="1y", interval="1d")
        trends["1D"] = _analyze_single_tf(df_1d, "1D (Setup)")
    except Exception as exc:
        logger.warning("Failed to fetch 1D data for %s: %s", symbol, exc)
        trends["1D"] = TimeframeTrend("1D (Setup)", "Neutral", False, 0, 0.0)

    # 3. Hourly Trend (Micro Entry)
    try:
        df_1h = fetch_ohlcv(symbol, period="1mo", interval="1h")
        trends["1H"] = _analyze_single_tf(df_1h, "1H (Micro)")
    except Exception as exc:
        logger.warning("Failed to fetch 1h data for %s: %s", symbol, exc)
        trends["1H"] = TimeframeTrend("1H (Micro)", "Neutral", False, 0, 0.0)

    # Calculate overall alignment
    scores = {"Bullish": 1.0, "Bearish": -1.0, "Neutral": 0.0}
    val_1w = scores[trends["1W"].trend]
    val_1d = scores[trends["1D"].trend]
    val_1h = scores[trends["1H"].trend]

    # Weighted alignment score (Macro: 40%, Setup: 40%, Micro: 20%)
    alignment_score = (val_1w * 0.40) + (val_1d * 0.40) + (val_1h * 0.20)

    if val_1w == 1.0 and val_1d == 1.0 and val_1h == 1.0:
        status = "Full Bullish Alignment (3/3)"
        modifier = 15.0
        desc = "Perfect multi-timeframe bullish confluence across Weekly, Daily & Hourly charts."
    elif val_1w == -1.0 and val_1d == -1.0 and val_1h == -1.0:
        status = "Full Bearish Alignment (3/3)"
        modifier = -15.0
        desc = "Strong multi-timeframe bearish alignment across Weekly, Daily & Hourly charts."
    elif val_1w == val_1d and val_1w != 0.0:
        status = f"Partial {trends['1W'].trend} Alignment (2/3)"
        modifier = 8.0 if val_1w == 1.0 else -8.0
        desc = f"Macro (1W) and Intermediate (1D) trends are aligned {trends['1W'].trend}."
    else:
        status = "Timeframe Divergence Detected"
        modifier = 0.0
        desc = "Conflicting signals across Weekly, Daily, and Hourly timeframes."

    return MTFResult(
        symbol=symbol,
        trends=trends,
        alignment_status=status,
        alignment_score=alignment_score,
        confidence_modifier=modifier,
        description=desc,
    )
