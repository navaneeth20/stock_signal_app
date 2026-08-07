"""
strategies/signal_engine.py
============================
Multi-confirmation signal generator.
Orchestrates all indicators → produces a final signal + confidence score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    EMA_LONG,
    EMA_MID,
    EMA_SHORT,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
    VWAP_PERIOD,
    VOLUME_MA_PERIOD,
)
from indicators.adx import adx_signal, compute_adx
from indicators.atr import compute_atr
from indicators.bollinger import compute_bollinger
from indicators.ema import compute_ema, ema_signal
from indicators.macd import compute_macd, macd_signal
from indicators.rsi import compute_rsi, rsi_signal
from indicators.supertrend import compute_supertrend, supertrend_signal
from indicators.vwap import compute_vwap, vwap_signal
from strategies.scoring import compute_score, label_from_score

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Complete trading signal result."""

    symbol: str
    signal: str               # Strong Buy | Buy | Hold | Sell | Strong Sell
    confidence: float         # 0–100 %
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasons: list[str] = field(default_factory=list)
    indicator_scores: dict = field(default_factory=dict)
    df: Optional[pd.DataFrame] = None
    signal_age_days: int = 1
    recommended_horizon: str = "5–15 Trading Days"
    mtf_result: Optional[Any] = None
    news_result: Optional[Any] = None
    mc_result: Optional[Any] = None
    pct_1w: float = 0.0
    pct_2w: float = 0.0
    pct_1m: float = 0.0
    dist_52w_high: float = 0.0
    is_extended: bool = False
    extended_warning: str = ""


def compute_price_performance(df: pd.DataFrame) -> dict:
    """
    Compute multi-period price performance metrics:
    - 1-Week % change (5 trading days)
    - 2-Week % change (10 trading days)
    - 1-Month % change (21 trading days)
    - Distance to 52-Week High (%)
    - Extended move / peak risk evaluation
    """
    if df.empty or len(df) < 5:
        return {
            "pct_1w": 0.0,
            "pct_2w": 0.0,
            "pct_1m": 0.0,
            "dist_52w_high": 0.0,
            "is_extended": False,
            "warning": "",
        }

    close = df["Close"]
    latest = float(close.iloc[-1])

    # 1-Week (5 trading days ago)
    idx_1w = max(0, len(df) - 6)
    p_1w = float(close.iloc[idx_1w])
    pct_1w = ((latest - p_1w) / p_1w) * 100.0 if p_1w > 0 else 0.0

    # 2-Week (10 trading days ago)
    idx_2w = max(0, len(df) - 11)
    p_2w = float(close.iloc[idx_2w])
    pct_2w = ((latest - p_2w) / p_2w) * 100.0 if p_2w > 0 else 0.0

    # 1-Month (21 trading days ago)
    idx_1m = max(0, len(df) - 22)
    p_1m = float(close.iloc[idx_1m])
    pct_1m = ((latest - p_1m) / p_1m) * 100.0 if p_1m > 0 else 0.0

    # 52-Week High Proximity (last 252 bars max)
    high_col = df["High"] if "High" in df.columns else close
    window_52w = high_col.iloc[-min(252, len(df)):]
    high_52w = float(window_52w.max())
    dist_52w_high = ((high_52w - latest) / high_52w) * 100.0 if high_52w > 0 else 0.0

    # Over-extended move evaluation
    is_extended = False
    warning = ""

    if pct_2w >= 10.0 or (pct_1w >= 6.0 and dist_52w_high <= 2.0):
        is_extended = True
        warning = f"Over-extended move warning: Stock is up +{pct_2w:.1f}% over the last 2 weeks ({dist_52w_high:.1f}% below 52W high). High risk of buying near peak."
    elif pct_2w >= 6.0:
        warning = f"Moderate rally: Stock has gained +{pct_2w:.1f}% in 2 weeks. Monitor for potential short-term resistance."

    return {
        "pct_1w": pct_1w,
        "pct_2w": pct_2w,
        "pct_1m": pct_1m,
        "dist_52w_high": dist_52w_high,
        "is_extended": is_extended,
        "warning": warning,
    }


def _compute_signal_age(df: pd.DataFrame) -> int:
    """Count consecutive bars the current trend signal has been active."""
    if "Supertrend_Direction" in df.columns and not df.empty:
        col = df["Supertrend_Direction"]
        last_val = col.iloc[-1]
        count = 0
        for val in reversed(col.tolist()):
            if val == last_val:
                count += 1
            else:
                break
        return max(1, count)
    return 1


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all indicator computations on the input OHLCV DataFrame.

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        DataFrame enriched with all indicator columns.
    """
    df = compute_ema(df, periods=[EMA_SHORT, EMA_MID, EMA_LONG])
    df = compute_rsi(df, period=RSI_PERIOD)
    df = compute_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df = compute_supertrend(df, period=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULTIPLIER)
    df = compute_adx(df, period=ADX_PERIOD)
    df = compute_atr(df, period=ATR_PERIOD)
    df = compute_bollinger(df, period=BB_PERIOD, std_dev=BB_STD)
    df = compute_vwap(df, period=VWAP_PERIOD)

    # Volume SMA
    df["Volume_SMA"] = df["Volume"].rolling(window=VOLUME_MA_PERIOD).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

    return df


def volume_signal(df: pd.DataFrame) -> dict:
    """Simple volume confirmation signal."""
    if "Volume_Ratio" not in df.columns:
        return {"signal": 0, "score": 0, "reasons": ["Volume SMA not available"]}

    ratio = df["Volume_Ratio"].iloc[-1]
    if ratio > 1.5:
        return {"signal": 1, "score": 1, "reasons": [f"Volume {ratio:.1f}x avg — Strong buying interest"]}
    elif ratio > 1.2:
        return {"signal": 1, "score": 1, "reasons": [f"Volume {ratio:.1f}x avg — Above-average activity"]}
    elif ratio < 0.5:
        return {"signal": -1, "score": -1, "reasons": [f"Volume {ratio:.1f}x avg — Very low interest"]}
    else:
        return {"signal": 0, "score": 0, "reasons": [f"Volume {ratio:.1f}x avg — Normal"]}


def generate_signal(symbol: str, df: pd.DataFrame) -> SignalResult:
    """
    Generate a complete trading signal for a stock.

    Args:
        symbol: Stock symbol (for labeling).
        df:     Enriched DataFrame (must have all indicator columns).

    Returns:
        SignalResult dataclass.
    """
    if len(df) < 60:
        raise ValueError(f"Insufficient data for {symbol}: need ≥60 bars, got {len(df)}")

    # Individual indicator signals
    ema_s = ema_signal(df)
    rsi_s = rsi_signal(df, overbought=RSI_OVERBOUGHT, oversold=RSI_OVERSOLD)
    macd_s = macd_signal(df)
    st_s = supertrend_signal(df)
    adx_s = adx_signal(df)
    vol_s = volume_signal(df)
    vwap_s = vwap_signal(df)

    indicator_signals = {
        "ema": ema_s,
        "rsi": rsi_s,
        "macd": macd_s,
        "supertrend": st_s,
        "adx": adx_s,
        "volume": vol_s,
        "vwap": vwap_s,
    }

    # Compute multi-period price performance & peak risk
    perf = compute_price_performance(df)

    # Score
    score_result = compute_score(indicator_signals)
    confidence = score_result["confidence"]
    signal_label = label_from_score(confidence)

    # Collect all reasons
    all_reasons: list[str] = []
    for key, sig in indicator_signals.items():
        all_reasons.extend(sig.get("reasons", []))

    # Over-extended move safeguard adjustment
    if perf["is_extended"] and signal_label in ("Strong Buy", "Buy"):
        confidence = max(0.0, confidence - 12.0)
        new_label = label_from_score(confidence)
        if new_label != signal_label:
            all_reasons.append(f"⚠️ Signal downgraded from {signal_label} to {new_label} due to extended 2-week runup (+{perf['pct_2w']:.1f}% near peak).")
            signal_label = new_label
        else:
            all_reasons.append(f"⚠️ Confidence adjusted (-12%) due to extended 2-week runup (+{perf['pct_2w']:.1f}% near peak).")

    # Risk levels (based on last ATR)
    last = df.iloc[-1]
    entry = float(last["Close"])
    raw_atr = last.get("ATR", entry * 0.02)
    atr = float(raw_atr) if (pd.notna(raw_atr) and float(raw_atr) > 0) else entry * 0.02
    stop_loss = max(0.01, entry - 1.5 * atr)
    take_profit = entry + 3.0 * atr  # 2:1 RRR at minimum
    risk_dist = max(entry - stop_loss, 0.01)
    rr = (take_profit - entry) / risk_dist

    signal_age = _compute_signal_age(df)
    horizon = "3–7 Trading Days" if signal_label in ("Strong Buy", "Strong Sell") else "7–15 Trading Days"

    logger.info("Signal for %s: %s (%.1f%%, %d days active, 2W %+.1f%%)", symbol, signal_label, confidence, signal_age, perf["pct_2w"])

    return SignalResult(
        symbol=symbol,
        signal=signal_label,
        confidence=confidence,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=rr,
        reasons=all_reasons,
        indicator_scores=score_result["scores"],
        df=df,
        signal_age_days=signal_age,
        recommended_horizon=horizon,
        pct_1w=perf["pct_1w"],
        pct_2w=perf["pct_2w"],
        pct_1m=perf["pct_1m"],
        dist_52w_high=perf["dist_52w_high"],
        is_extended=perf["is_extended"],
        extended_warning=perf["warning"],
    )


