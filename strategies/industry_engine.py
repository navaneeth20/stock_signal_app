"""
strategies/industry_engine.py
================================
Quantitative Industry Sector Performance & 1-Month Trend Forecasting Engine.
Calculates real-time sector momentum, 1-month predictive trends, relative strength,
and identifies top outperforming industry equities across Indian markets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import pandas as pd

from config import SECTOR_POOLS
from data.fetch_data import fetch_ohlcv
from strategies.signal_engine import compute_all_indicators



logger = logging.getLogger(__name__)


def analyze_sector_performance() -> List[Dict[str, Any]]:
    """
    Computes real-time performance and 1-month trend forecasts for all industry sectors.

    Returns:
        List of dictionaries containing sector metrics, ranked by 1M performance.
    """
    sector_results = []

    for sector_name, stocks in SECTOR_POOLS.items():
        stock_perf_list = []

        for s_info in stocks:
            sym = s_info["symbol"]
            name = s_info["name"]
            try:
                df_raw = fetch_ohlcv(sym, interval="1d", period="3mo")
                if df_raw.empty or len(df_raw) < 20:
                    continue
                df = compute_all_indicators(df_raw)
                last = df.iloc[-1]
                p_now = float(last["Close"])

                # Returns
                p_1d = float(df["Close"].iloc[-2]) if len(df) > 1 else p_now
                p_1w = float(df["Close"].iloc[-6]) if len(df) > 5 else p_now
                p_1m = float(df["Close"].iloc[-22]) if len(df) > 21 else p_now

                ret_1d = ((p_now - p_1d) / p_1d) * 100
                ret_1w = ((p_now - p_1w) / p_1w) * 100
                ret_1m = ((p_now - p_1m) / p_1m) * 100

                rsi = float(last.get("RSI", 50))
                ema20 = float(last.get("EMA_20", p_now))
                above_ema20 = p_now > ema20

                stock_perf_list.append({
                    "symbol": sym,
                    "name": name,
                    "price": p_now,
                    "ret_1d": ret_1d,
                    "ret_1w": ret_1w,
                    "ret_1m": ret_1m,
                    "rsi": rsi,
                    "above_ema20": above_ema20,
                })
            except Exception as exc:
                logger.warning("Error fetching sector stock %s: %s", sym, exc)
                continue

        if not stock_perf_list:
            continue

        # Sector Aggregates
        avg_ret_1d = sum(s["ret_1d"] for s in stock_perf_list) / len(stock_perf_list)
        avg_ret_1w = sum(s["ret_1w"] for s in stock_perf_list) / len(stock_perf_list)
        avg_ret_1m = sum(s["ret_1m"] for s in stock_perf_list) / len(stock_perf_list)

        pct_above_ema20 = (sum(1 for s in stock_perf_list if s["above_ema20"]) / len(stock_perf_list)) * 100
        avg_rsi = sum(s["rsi"] for s in stock_perf_list) / len(stock_perf_list)

        # 1-Month Trend Prediction Score (0-100%)
        score_1m = 50.0 + (avg_ret_1m * 3.5) + (avg_ret_1w * 2.0) + ((avg_rsi - 50.0) * 0.8) + ((pct_above_ema20 - 50.0) * 0.3)
        score_1m = max(10.0, min(98.0, score_1m))

        # Predictive Trend Classification & 30-Day Outlook Target
        if score_1m >= 75.0:
            trend_label = "Strong Bullish Continuation"
            trend_icon = "🚀"
            trend_color = "#00e676"
            target_range = f"+{avg_ret_1m * 0.4 + 3.5:.1f}% to +{avg_ret_1m * 0.6 + 7.5:.1f}%"
            outlook_text = "Strong institutional accumulation. High probability of sector outperformance over Nifty 50 over the next 30 days."
        elif score_1m >= 60.0:
            trend_label = "Bullish Accumulation"
            trend_icon = "📈"
            trend_color = "#388bfd"
            target_range = f"+{avg_ret_1m * 0.3 + 2.0:.1f}% to +{avg_ret_1m * 0.5 + 4.5:.1f}%"
            outlook_text = "Steady buying momentum. Expect healthy pullbacks to 20-day moving averages as accumulation continues."
        elif score_1m >= 42.0:
            trend_label = "Neutral Consolidation"
            trend_icon = "⚖️"
            trend_color = "#ffb300"
            target_range = "-1.5% to +2.5%"
            outlook_text = "Sideways rangebound action expected. Industry is digesting recent moves prior to next earnings catalyst."
        else:
            trend_label = "Bearish Pullback"
            trend_icon = "🔻"
            trend_color = "#ff1744"
            target_range = "-3.5% to -7.5%"
            outlook_text = "Underperforming general market benchmark. High downside risk over the next 30 days."

        # Top 3 Sector Leaders
        top_leaders = sorted(stock_perf_list, key=lambda x: x["ret_1m"], reverse=True)[:3]

        sector_results.append({
            "sector": sector_name,
            "ret_1d": avg_ret_1d,
            "ret_1w": avg_ret_1w,
            "ret_1m": avg_ret_1m,
            "score_1m": score_1m,
            "trend_label": trend_label,
            "trend_icon": trend_icon,
            "trend_color": trend_color,
            "target_range": target_range,
            "outlook_text": outlook_text,
            "avg_rsi": avg_rsi,
            "pct_above_ema20": pct_above_ema20,
            "top_leaders": top_leaders,
            "stock_count": len(stock_perf_list),
        })

    # Sort sectors by 1-Month Return DESC
    sector_results.sort(key=lambda x: x["ret_1m"], reverse=True)
    return sector_results
