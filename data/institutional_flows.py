"""
data/institutional_flows.py
============================
Market-cap classification plus a **price/volume accumulation proxy**.

Read this before touching the module:

This is NOT shareholding data, and nothing here should ever be labelled FII,
DII or mutual-fund holdings. Real Indian institutional ownership comes from
sources this app does not fetch:

  * BSE/NSE quarterly shareholding pattern filings (promoter / FII / DII /
    public splits, filed within 21 days of quarter end)
  * AMFI and AMC monthly portfolio disclosures (fund-level MF holdings)
  * NSE/BSE bulk and block deal feeds (large single transactions)

yfinance exposes ``heldPercentInstitutions`` and ``heldPercentInsiders``, but
for Indian tickers these are frequently absent, stale, or simply wrong — and
they carry no FII/DII breakdown at all. The previous version of this module
invented that breakdown with a hardcoded 52/48 split, derived a "QoQ net
buying %" from the last month's price change, converted it to a rupee-crore
flow figure, and returned the same four fund names for every stock in the
market. All of that is gone.

What remains is honest: an accumulation proxy built from price and volume that
says only what it can observe, and is labelled as such everywhere it renders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import INSTITUTIONAL_PROXY_DISCLAIMER

logger = logging.getLogger(__name__)


@dataclass
class AccumulationProxyResult:
    """
    Price/volume-derived accumulation reading. Not shareholding data.

    Every field below is computed from the OHLCV series. Ownership fields are
    populated only when yfinance actually returns them; otherwise they are
    None and the UI must show "Not available" rather than a default.
    """

    symbol: str

    # Market cap (from yfinance; None when unavailable)
    mcap_cr: Optional[float]
    mcap_category: str
    mcap_badge_color: str

    # Reported ownership — only what the source actually gave us.
    total_institutional_pct: Optional[float] = None
    insider_pct: Optional[float] = None
    ownership_source: str = "Not available"
    ownership_available: bool = False

    # Accumulation proxy (price + volume behaviour)
    accumulation_score: float = 0.0          # -100 .. +100
    accumulation_label: str = "Neutral"
    accumulation_color: str = "var(--ink-2)"
    up_volume_share_pct: float = 50.0        # % of 30d volume on up days
    obv_slope_pct: float = 0.0               # OBV trend over the window
    close_location_value: float = 0.0        # -1 .. +1, where closes sit in range
    ret_1m_pct: float = 0.0
    ret_1w_pct: float = 0.0

    evidence: List[str] = field(default_factory=list)
    disclaimer: str = INSTITUTIONAL_PROXY_DISCLAIMER


def _classify_mcap(mcap_cr: Optional[float]) -> tuple[str, str]:
    """SEBI-style market cap bucket. Returns (label, colour)."""
    # Colour is reserved for buy/sell state, so every tier returns the same
    # neutral token; the label itself carries the meaning.
    if mcap_cr is None:
        return "Market cap unavailable", "var(--ink-2)"
    if mcap_cr >= 20000.0:
        return "Large Cap", "var(--ink-2)"
    if mcap_cr >= 5000.0:
        return "Mid Cap", "var(--ink-2)"
    return "Small Cap", "var(--ink-2)"


def _safe_pct(value: Optional[float]) -> Optional[float]:
    """Convert a 0-1 fraction to a 0-100 percentage, or None."""
    if value is None:
        return None
    try:
        pct = float(value) * 100.0
    except (TypeError, ValueError):
        return None
    if not np.isfinite(pct) or pct < 0 or pct > 100:
        return None
    return round(pct, 2)


def fetch_accumulation_proxy(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    window: int = 30,
) -> AccumulationProxyResult:
    """
    Compute a price/volume accumulation proxy for a symbol.

    The proxy blends three observable things, none of which require ownership
    data:

      1. **Up-volume share** — what fraction of the window's volume traded on
         up days. Sustained buying shows up as volume concentrating on
         advances.
      2. **OBV slope** — on-balance volume trend, normalised. Rising OBV while
         price consolidates is the classic accumulation footprint.
      3. **Close location value** — where closes land inside each bar's range.
         Persistent closes near the high suggest demand absorbing supply.

    Args:
        symbol: Ticker symbol (e.g. RELIANCE.NS).
        df:     Enriched OHLCV DataFrame. Required for the proxy.
        window: Lookback in trading days.

    Returns:
        AccumulationProxyResult.
    """
    clean_sym = symbol.upper().strip()

    # ── Market cap and whatever ownership the source will admit to ──────────
    mcap_cr: Optional[float] = None
    total_inst_pct: Optional[float] = None
    insider_pct: Optional[float] = None
    ownership_source = "Not available"

    try:
        info = yf.Ticker(clean_sym).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch yfinance info for %s: %s", clean_sym, exc)
        info = {}

    mcap_raw = info.get("marketCap")
    if mcap_raw:
        try:
            mcap_cr = round(float(mcap_raw) / 1e7, 1)  # 1 crore = 1e7
        except (TypeError, ValueError):
            mcap_cr = None

    # No invented defaults. If the field is missing, it stays None.
    total_inst_pct = _safe_pct(info.get("heldPercentInstitutions"))
    insider_pct = _safe_pct(info.get("heldPercentInsiders"))
    if total_inst_pct is not None or insider_pct is not None:
        ownership_source = (
            "Yahoo Finance aggregate — no FII/DII breakdown, often stale for "
            "Indian listings. Verify against the latest BSE/NSE shareholding filing."
        )

    mcap_cat, mcap_color = _classify_mcap(mcap_cr)

    result = AccumulationProxyResult(
        symbol=clean_sym,
        mcap_cr=mcap_cr,
        mcap_category=mcap_cat,
        mcap_badge_color=mcap_color,
        total_institutional_pct=total_inst_pct,
        insider_pct=insider_pct,
        ownership_source=ownership_source,
        ownership_available=(total_inst_pct is not None or insider_pct is not None),
    )

    # ── The proxy itself ────────────────────────────────────────────────────
    if df is None or df.empty or len(df) < 10:
        result.evidence.append("Not enough price history to compute an accumulation reading.")
        return result

    recent = df.tail(min(window, len(df)))
    close = recent["Close"].astype(float)
    volume = recent["Volume"].astype(float) if "Volume" in recent.columns else None
    high = recent["High"].astype(float) if "High" in recent.columns else close
    low = recent["Low"].astype(float) if "Low" in recent.columns else close

    evidence: list[str] = []
    components: list[float] = []

    # 1. Up-volume share
    up_share = 50.0
    if volume is not None and volume.sum() > 0:
        change = close.diff()
        up_vol = float(volume[change > 0].sum())
        down_vol = float(volume[change < 0].sum())
        traded = up_vol + down_vol
        if traded > 0:
            up_share = (up_vol / traded) * 100.0
            components.append((up_share - 50.0) * 2.0)  # → -100..+100
            evidence.append(
                f"{up_share:.0f}% of the last {len(recent)} days' volume traded on up days"
            )

    # 2. OBV slope, normalised by total volume
    obv_slope = 0.0
    if volume is not None and volume.sum() > 0:
        direction = np.sign(close.diff().fillna(0.0))
        obv = (direction * volume).cumsum()
        span = float(obv.iloc[-1] - obv.iloc[0])
        denom = float(volume.sum())
        if denom > 0:
            obv_slope = (span / denom) * 100.0
            components.append(float(np.clip(obv_slope * 2.0, -100, 100)))
            trend = "rising" if obv_slope > 2 else ("falling" if obv_slope < -2 else "flat")
            evidence.append(f"On-balance volume is {trend} ({obv_slope:+.1f}% of window volume)")

    # 3. Close location value
    clv = 0.0
    rng = (high - low).replace(0, np.nan)
    clv_series = ((close - low) - (high - close)) / rng
    clv_clean = clv_series.dropna()
    if not clv_clean.empty:
        clv = float(clv_clean.mean())
        components.append(float(np.clip(clv * 100.0, -100, 100)))
        where = "upper" if clv > 0.15 else ("lower" if clv < -0.15 else "middle")
        evidence.append(f"Closes cluster in the {where} of each day's range (CLV {clv:+.2f})")

    ret_1w = 0.0
    ret_1m = 0.0
    full_close = df["Close"].astype(float)
    if len(full_close) > 5:
        ret_1w = float((full_close.iloc[-1] / full_close.iloc[-6] - 1) * 100)
    if len(full_close) > 21:
        ret_1m = float((full_close.iloc[-1] / full_close.iloc[-22] - 1) * 100)

    score = float(np.mean(components)) if components else 0.0
    score = float(np.clip(score, -100.0, 100.0))

    if score >= 40:
        label, color = "Strong accumulation footprint", "var(--pos)"
    elif score >= 15:
        label, color = "Mild accumulation footprint", "var(--accent)"
    elif score <= -40:
        label, color = "Strong distribution footprint", "var(--neg)"
    elif score <= -15:
        label, color = "Mild distribution footprint", "var(--neg)"
    else:
        label, color = "No clear footprint", "var(--ink-2)"

    result.accumulation_score = round(score, 1)
    result.accumulation_label = label
    result.accumulation_color = color
    result.up_volume_share_pct = round(up_share, 1)
    result.obv_slope_pct = round(obv_slope, 2)
    result.close_location_value = round(clv, 3)
    result.ret_1w_pct = round(ret_1w, 2)
    result.ret_1m_pct = round(ret_1m, 2)
    result.evidence = evidence

    return result


# Backwards-compatible alias. The old name promised data this module never had.
def fetch_institutional_flows(
    symbol: str, df: Optional[pd.DataFrame] = None
) -> AccumulationProxyResult:
    """Deprecated: use :func:`fetch_accumulation_proxy`."""
    logger.warning(
        "fetch_institutional_flows() is deprecated and returns a price/volume "
        "proxy, not shareholding data. Use fetch_accumulation_proxy()."
    )
    return fetch_accumulation_proxy(symbol, df)
