"""
strategies/signal_engine.py
============================
Multi-confirmation signal generator.
Orchestrates all indicators → produces a final signal + confidence score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

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
from strategies.risk import calculate_risk
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
    inst_result: Optional[Any] = None
    pct_1w: float = 0.0
    pct_2w: float = 0.0
    pct_1m: float = 0.0
    dist_52w_high: float = 0.0
    is_extended: bool = False
    extended_warning: str = ""


def apply_confidence_modifier(
    result: "SignalResult",
    delta: float,
    label: str,
    detail: str = "",
) -> "SignalResult":
    """
    Adjust a signal's confidence and re-derive its label in one step.

    The label is a pure function of the confidence, so anything that moves the
    confidence after ``generate_signal()`` must re-run ``label_from_score()``.
    Skipping that is how the UI ends up showing "Buy — 82%" when the Strong Buy
    threshold is 75. Always route post-hoc adjustments through here.

    Args:
        result: The SignalResult to adjust in place.
        delta:  Confidence points to add (may be negative). No-op when 0.
        label:  Short name of the adjustment, e.g. "Multi-timeframe alignment".
        detail: Optional extra context appended to the reason line.

    Returns:
        The same SignalResult, mutated.
    """
    if not delta:
        return result

    previous_label = result.signal
    result.confidence = max(0.0, min(100.0, result.confidence + delta))
    result.signal = label_from_score(result.confidence)

    suffix = f" {detail}" if detail else ""
    if result.signal != previous_label:
        result.reasons.append(
            f"{label}: confidence {delta:+.1f} → {result.confidence:.1f}%. "
            f"Signal revised from {previous_label} to {result.signal}.{suffix}"
        )
    else:
        result.reasons.append(
            f"{label}: confidence {delta:+.1f} → {result.confidence:.1f}%.{suffix}"
        )
    return result


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
    """
    Volume confirmation signal, signed by the bar's own direction.

    Volume has no direction of its own — it confirms whatever the candle did.
    Heavy volume on a down bar is distribution (bearish), not "strong buying
    interest". The magnitude comes from the volume ratio; the sign comes from
    the close-vs-open move.
    """
    if "Volume_Ratio" not in df.columns:
        return {"signal": 0, "score": 0, "reasons": ["Volume SMA not available"]}

    last = df.iloc[-1]
    ratio = last["Volume_Ratio"]
    if pd.isna(ratio):
        return {"signal": 0, "score": 0, "reasons": ["Volume SMA not available"]}

    ratio = float(ratio)
    close = float(last["Close"])
    open_px = float(last["Open"]) if "Open" in df.columns and pd.notna(last["Open"]) else close

    # Direction of the bar the volume belongs to.
    if close > open_px:
        direction, word = 1, "accumulation"
    elif close < open_px:
        direction, word = -1, "distribution"
    else:
        direction, word = 0, "indecision"

    if ratio > 1.5:
        magnitude = 2
        strength = "Heavy"
    elif ratio > 1.2:
        magnitude = 1
        strength = "Above-average"
    elif ratio < 0.5:
        # Thin volume conviction-free: fade whatever the bar did.
        return {
            "signal": 0,
            "score": 0,
            "reasons": [f"Volume {ratio:.1f}x avg — Very low participation, move lacks conviction"],
        }
    else:
        return {"signal": 0, "score": 0, "reasons": [f"Volume {ratio:.1f}x avg — Normal"]}

    if direction == 0:
        return {
            "signal": 0,
            "score": 0,
            "reasons": [f"Volume {ratio:.1f}x avg on a flat close — {word}"],
        }

    score = magnitude * direction
    move_pct = ((close - open_px) / open_px * 100.0) if open_px else 0.0
    return {
        "signal": 1 if score > 0 else -1,
        "score": score,
        "reasons": [
            f"Volume {ratio:.1f}x avg on a {move_pct:+.1f}% bar — {strength} {word}"
        ],
    }


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

    # Individual indicator signals.
    # ADX runs first so RSI can use it to decide between the momentum reading
    # (trending market) and the mean-reversion reading (ranging market).
    adx_s = adx_signal(df)
    ema_s = ema_signal(df)
    rsi_s = rsi_signal(
        df,
        overbought=RSI_OVERBOUGHT,
        oversold=RSI_OVERSOLD,
        adx_value=adx_s.get("adx_value"),
    )
    macd_s = macd_signal(df)
    st_s = supertrend_signal(df)
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

    # Risk levels — single source of truth in strategies/risk.py, which honours
    # ATR_STOP_MULTIPLIER / TAKE_PROFIT_RR from config and gets the direction
    # right for short signals (stop above entry, target below).
    risk = calculate_risk(df, signal_label)
    entry = risk.entry_price
    stop_loss = risk.stop_loss
    take_profit = risk.take_profit
    rr = risk.risk_reward

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


