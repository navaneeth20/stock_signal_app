"""
app.py
======
StockSense AI — AI-Powered Indian Stock Market Signal Dashboard
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    AI_MAX_TOKENS,
    AI_MODEL,
    BT_DEFAULT_CAPITAL,
    DEFAULT_CAPITAL,
    DEFAULT_INTERVAL,
    DEFAULT_RISK_PER_TRADE,
    INDEX_GROUPS,
    NIFTY50_STOCKS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SIGNAL_COLORS,
    SIGNAL_EMOJI,
    SUPPORTED_INTERVALS,
)
from alerts.email import format_signal_email, send_email_alert
from alerts.telegram import format_signal_message, send_telegram_alert
from backtesting.backtest import run_backtest
from charts.candlestick import build_equity_curve, build_price_chart
from data.fetch_data import (
    fetch_ohlcv,
    get_company_info,
    get_sector_peers,
    get_stock_name,
    normalise_symbol,
)
from data.news_sentiment import fetch_news_sentiment

from database.database import (
    add_to_watchlist,
    get_recent_signals,
    get_watchlist,
    initialise_db,
    is_in_watchlist,
    remove_from_watchlist,
    save_signal,
)
from indicators.mtf import compute_mtf_alignment
from strategies.risk import calculate_risk
from strategies.signal_engine import compute_all_indicators, generate_signal
from utils.helpers import color_for_signal, format_inr, format_volume, pct_change
from utils.quant_risk import run_monte_carlo_simulation


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/",
        "About": f"**{APP_NAME} v{APP_VERSION}** — {APP_TAGLINE}",
    },
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Theme & Background ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background: radial-gradient(circle at 50% 0%, #151d2a 0%, #090c10 70%, #05070a 100%);
    color: #e6edf3;
}

/* ── Typography & Ticker Fonts ── */
.mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Hero Terminal Header Banner ── */
.hero-header {
    background: linear-gradient(135deg, rgba(22, 27, 34, 0.75) 0%, rgba(13, 17, 23, 0.9) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(88, 166, 255, 0.18);
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 28px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #00e676, #00f2fe);
    background-size: 300% 100%;
    animation: gradientShift 6s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 230, 118, 0.1);
    border: 1px solid rgba(0, 230, 118, 0.3);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    color: #00e676;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pulsing-dot {
    width: 8px;
    height: 8px;
    background-color: #00e676;
    border-radius: 50%;
    box-shadow: 0 0 10px #00e676;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(0, 230, 118, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
}

/* ── Metric Glass Cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(22, 27, 34, 0.65) 0%, rgba(13, 17, 23, 0.75) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(88, 166, 255, 0.4);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(88, 166, 255, 0.15);
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #f0f6fc;
}

/* ── Signal Hero Card ── */
.hero-signal-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(9, 14, 23, 0.95) 100%);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    position: relative;
}

/* ── Section headers ── */
.section-header {
    font-size: 16px;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 0.04em;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(88, 166, 255, 0.15);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── AI Explanation Box ── */
.ai-box {
    background: linear-gradient(135deg, rgba(13, 33, 55, 0.7) 0%, rgba(10, 22, 40, 0.85) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 16px;
    padding: 24px;
    font-size: 14.5px;
    line-height: 1.75;
    color: #c9d1d9;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: rgba(13, 17, 23, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* ── Modern Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(13, 17, 23, 0.6);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 13.5px;
    color: #8b949e;
    border: none !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e6edf3;
    background: rgba(255, 255, 255, 0.05);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(31, 111, 235, 0.4) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.25s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    border-color: #3fb950 !important;
    box-shadow: 0 6px 20px rgba(46, 160, 67, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(46, 160, 67, 0.5) !important;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080b10; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

/* ── Mobile & Responsive UI Media Queries ── */
@media (max-width: 768px) {
    .hero-header {
        padding: 16px 14px !important;
        margin-bottom: 16px !important;
        border-radius: 14px !important;
    }
    .hero-header h1 {
        font-size: 20px !important;
    }
    .hero-signal-card {
        padding: 20px 14px !important;
        border-radius: 14px !important;
    }
    .hero-signal-card div[style*="font-size:42px"] {
        font-size: 26px !important;
    }
    .hero-signal-card div[style*="width:65%"] {
        width: 92% !important;
    }
    .metric-card {
        padding: 12px 10px !important;
        margin-bottom: 8px !important;
    }
    .metric-value {
        font-size: 16px !important;
    }
    .metric-label {
        font-size: 10px !important;
    }
    /* Horizontally scrollable tabs on mobile screens */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        white-space: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        gap: 6px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px !important;
        font-size: 12px !important;
    }
    [data-testid="column"] {
        min-width: 48% !important;
        flex: 1 1 45% !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ────────────────────────────────────────────────────────
def _init_session() -> None:
    defaults = {
        "df": None,
        "signal_result": None,
        "selected_symbol": "RELIANCE.NS",
        "auto_refresh": False,
        "last_refresh": 0.0,
        "scanner_results": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()
initialise_db()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo + title
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, rgba(31,111,235,0.15), rgba(88,166,255,0.05)); border:1px solid rgba(88,166,255,0.2); border-radius:16px; padding:20px 16px; text-align:center; margin-bottom:20px; box-shadow:0 8px 24px rgba(0,0,0,0.3);">
            <div style="background:linear-gradient(135deg,#1f6feb,#58a6ff); width:48px; height:48px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center; font-size:24px; box-shadow:0 6px 20px rgba(31,111,235,0.4); margin-bottom:10px;">📈</div>
            <div style="font-size:18px; font-weight:800; color:#f0f6fc; letter-spacing:0.02em;">{APP_NAME}</div>
            <div style="font-size:11px; font-weight:500; color:#8b949e; margin-top:4px;">{APP_TAGLINE}</div>
            <div class="mono-font" style="display:inline-block; font-size:10px; font-weight:600; color:#58a6ff; background:rgba(88,166,255,0.1); padding:2px 8px; border-radius:10px; margin-top:8px;">v{APP_VERSION} PRO</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Stock Selection ──────────────────────────────────────────────────────
    st.markdown("#### 🏦 Stock Selection")

    # Build symbol→name mapping for display
    all_stocks = NIFTY50_STOCKS
    symbol_map = {s["symbol"]: f"{s['name']} ({s['symbol']})" for s in all_stocks}

    search_input = st.text_input(
        "Custom Symbol",
        placeholder="e.g. WIPRO, TATAMOTORS",
        help="Enter any NSE symbol without .NS suffix",
        key="custom_symbol_input",
    )
    if search_input.strip():
        custom_sym = normalise_symbol(search_input.strip(), "NSE")
        selected_symbol = custom_sym
    else:
        selected_display = st.selectbox(
            "Select Stock",
            options=list(symbol_map.keys()),
            format_func=lambda s: symbol_map[s],
            index=0,
        )
        selected_symbol = selected_display

    exchange = st.selectbox("Exchange", ["NSE", "BSE"])

    st.divider()

    # ── Timeframe ────────────────────────────────────────────────────────────
    st.markdown("#### ⏱ Timeframe")
    interval_label = st.selectbox("Interval", list(SUPPORTED_INTERVALS.keys()), index=0)
    interval = SUPPORTED_INTERVALS[interval_label]

    use_custom_dates = st.checkbox("Custom Date Range", value=False)
    if use_custom_dates:
        col_d1, col_d2 = st.columns(2)
        start_date = col_d1.date_input("From", datetime.now() - timedelta(days=365))
        end_date = col_d2.date_input("To", datetime.now())
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
    else:
        period_label = st.selectbox(
            "Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
            index=3,
        )
        period_map = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
        }
        period = period_map[period_label]
        start_dt = None
        end_dt = None

    st.divider()

    # ── Indicators Toggle ────────────────────────────────────────────────────
    st.markdown("#### 📊 Chart Overlays")
    show_ema = st.checkbox("EMA (20/50/200)", value=True)
    show_supertrend = st.checkbox("Supertrend", value=True)
    show_bollinger = st.checkbox("Bollinger Bands", value=True)
    show_vwap = st.checkbox("VWAP", value=True)
    show_volume = st.checkbox("Volume", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)

    st.divider()

    # ── Risk Settings ────────────────────────────────────────────────────────
    st.markdown("#### 💰 Risk Settings")
    capital = st.number_input(
        "Capital (₹)",
        min_value=10_000,
        max_value=10_000_000,
        value=DEFAULT_CAPITAL,
        step=10_000,
        format="%d",
    )
    risk_pct = st.slider("Risk Per Trade (%)", 0.5, 5.0, DEFAULT_RISK_PER_TRADE * 100, 0.5)

    st.divider()

    # ── Auto Refresh ─────────────────────────────────────────────────────────
    st.markdown("#### 🔄 Auto Refresh")
    auto_refresh = st.toggle("Enable Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh every (seconds)", 30, 600, 60, 30)

    # ── Load Button ──────────────────────────────────────────────────────────
    st.divider()
    load_clicked = st.button("🚀 Analyse Stock", type="primary", use_container_width=True)

    # ── Alerts Config ────────────────────────────────────────────────────────
    with st.expander("🔔 Alert Settings"):
        tg_token = st.text_input("Telegram Bot Token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
        tg_chat = st.text_input("Telegram Chat ID", value=os.getenv("TELEGRAM_CHAT_ID", ""))
        email_from = st.text_input("Email (From)", value=os.getenv("EMAIL_SENDER", ""))
        email_pass = st.text_input("Email Password", type="password")
        email_to = st.text_input("Email (To)", value=os.getenv("EMAIL_RECEIVER", ""))

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Load & Analyse
# ═══════════════════════════════════════════════════════════════════════════════

def _find_better_alternatives(current_symbol: str, current_confidence: float) -> tuple[str, list[dict]]:
    """Scan sector/category peer stocks for higher confidence or stronger signals."""
    sector_name, peer_pool = get_sector_peers(current_symbol)
    better_list = []
    for item in peer_pool:
        sym = item["symbol"]
        if sym.upper() == current_symbol.upper():
            continue
        try:
            raw = fetch_ohlcv(sym, period="1y", interval="1d")
            if raw.empty or len(raw) < 60:
                continue
            d = compute_all_indicators(raw)
            res = generate_signal(sym, d)
            if res.confidence >= current_confidence or (res.signal in ("Strong Buy", "Buy") and res.confidence >= 60):
                last_p = d["Close"].iloc[-1]
                prev_p = d["Close"].iloc[-2] if len(d) > 1 else last_p
                chg = pct_change(float(prev_p), float(last_p))
                better_list.append({
                    "symbol": sym,
                    "name": item["name"],
                    "signal": res.signal,
                    "confidence": res.confidence,
                    "price": last_p,
                    "change": chg,
                    "rr": res.risk_reward,
                    "sector": sector_name,
                })
        except Exception:
            continue
    better_list.sort(key=lambda x: x["confidence"], reverse=True)
    return sector_name, better_list[:3]


def load_and_analyse(symbol: str) -> None:
    """Fetch data and run full signal analysis, store in session state."""
    with st.spinner(f"Fetching & Analysing {symbol}…"):
        try:
            if use_custom_dates:
                df_raw = fetch_ohlcv(symbol, interval=interval, start=start_dt, end=end_dt)
            else:
                df_raw = fetch_ohlcv(symbol, interval=interval, period=period)

            df = compute_all_indicators(df_raw)
            result = generate_signal(symbol, df)
            risk = calculate_risk(df, result.signal, capital=capital, risk_per_trade=risk_pct / 100)

            # Get full official company name
            comp_name = get_stock_name(symbol)

            # 1. Multi-Timeframe (MTF) Alignment Analysis
            mtf_res = compute_mtf_alignment(symbol)
            result.mtf_result = mtf_res
            # Apply MTF confidence modifier
            result.confidence = max(0.0, min(100.0, result.confidence + mtf_res.confidence_modifier))

            # 2. Market Sentiment & News Intelligence
            news_res = fetch_news_sentiment(symbol, comp_name)
            result.news_result = news_res

            # 3. Quantitative Risk & Monte Carlo Simulation
            mc_res = run_monte_carlo_simulation(df, risk.entry_price, risk.stop_loss, risk.take_profit)
            result.mc_result = mc_res

            # Find better sector/category alternatives
            sector_category, alternatives = _find_better_alternatives(symbol, result.confidence)

            # Persist signal
            save_signal(
                symbol,
                result.signal,
                result.confidence,
                risk.entry_price,
                risk.stop_loss,
                risk.take_profit,
                risk.risk_reward,
                interval,
            )

            st.session_state.df = df
            st.session_state.signal_result = result
            st.session_state.risk = risk
            st.session_state.company_name = comp_name
            st.session_state.sector_category = sector_category
            st.session_state.better_alternatives = alternatives
            st.session_state.selected_symbol = symbol
            st.session_state.last_refresh = time.time()
            logger.info("Analysis complete for %s (%s): %s", symbol, comp_name, result.signal)

        except Exception as exc:
            st.error(f"❌ Error loading {symbol}: {exc}")
            logger.exception("load_and_analyse failed for %s", symbol)




# Auto-refresh logic
if auto_refresh:
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed > refresh_interval:
        load_and_analyse(selected_symbol)

# Trigger on button click
if load_clicked:
    load_and_analyse(selected_symbol)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="hero-header">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div style="display:flex; align-items:center; gap:18px;">
                <div style="background:linear-gradient(135deg,#1f6feb,#58a6ff); width:52px; height:52px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:26px; box-shadow:0 8px 24px rgba(31,111,235,0.4);">📈</div>
                <div>
                    <div style="display:flex; align-items:center; gap:12px;">
                        <h1 style="margin:0; font-size:28px; font-weight:800; background:linear-gradient(90deg,#58a6ff,#00e5ff,#79c0ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.02em;">{APP_NAME} <span style="font-size:13px; font-weight:700; padding:3px 10px; border-radius:20px; background:rgba(88,166,255,0.12); color:#58a6ff; border:1px solid rgba(88,166,255,0.25); -webkit-text-fill-color:initial;">PRO TERMINAL</span></h1>
                    </div>
                    <p style="margin:4px 0 0; color:#8b949e; font-size:13.5px; font-weight:500;">{APP_TAGLINE}</p>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:14px;">
                <div class="live-indicator"><span class="pulsing-dot"></span> LIVE NSE / BSE</div>
                <div class="mono-font" style="font-size:12px; color:#8b949e; background:rgba(255,255,255,0.04); padding:6px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">v{APP_VERSION}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — AI Explanation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_ai_explanation(result, risk) -> str:
    """
    Generate a natural-language trade explanation.
    Uses OpenAI API if configured, otherwise generates a rule-based explanation.
    """
    sig_emoji = SIGNAL_EMOJI.get(result.signal, "📊")

    # Try OpenAI API
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            prompt = (
                f"You are a professional Indian stock market analyst. "
                f"Generate a concise 5-7 sentence trading explanation for the following signal:\n\n"
                f"Stock: {result.symbol}\n"
                f"Signal: {result.signal} ({result.confidence:.1f}% confidence)\n"
                f"Entry: ₹{risk.entry_price:.2f}, Stop Loss: ₹{risk.stop_loss:.2f}, "
                f"Target: ₹{risk.take_profit:.2f}, RR: 1:{risk.risk_reward:.1f}\n"
                f"Key reasons: {'; '.join(result.reasons[:5])}\n\n"
                "Be professional, concise, and mention risk management."
            )
            resp = client.chat.completions.create(
                model=AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=AI_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("OpenAI call failed: %s", exc)

    # Rule-based fallback
    trend = (
        "bullish" if result.signal in ("Strong Buy", "Buy")
        else "bearish" if result.signal in ("Sell", "Strong Sell")
        else "neutral"
    )
    lines = [
        f"<b>{sig_emoji} {result.signal} — {result.confidence:.1f}% Confidence</b><br><br>",
        f"The overall trend for <b>{result.symbol}</b> is currently <b>{trend}</b>. ",
    ]
    for r in result.reasons[:4]:
        lines.append(f"{r}. ")
    lines.append(
        f"<br><br><b>Risk Management:</b> Entry at <b>₹{risk.entry_price:,.2f}</b>, "
        f"stop loss at <b>₹{risk.stop_loss:,.2f}</b> ({risk.stop_pct:.1f}% below entry), "
        f"and target at <b>₹{risk.take_profit:,.2f}</b> gives a risk-to-reward ratio of "
        f"<b>1:{risk.risk_reward:.1f}</b>. "
    )
    lines.append(
        "<br><br><i>⚠️ This analysis is for educational purposes only. "
        "Always manage your risk and trade responsibly.</i>"
    )
    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_signal, tab_chart, tab_backtest, tab_scanner, tab_watchlist, tab_alerts = st.tabs(
    ["🎯 Signal", "📊 Chart", "⚡ Backtest", "🔍 Scanner", "⭐ Watchlist", "🔔 Alerts"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

with tab_signal:
    if st.session_state.signal_result is None:
        st.markdown(
            """
            <div style="text-align:center;padding:80px 20px;">
                <div style="font-size:64px;margin-bottom:20px;">🚀</div>
                <h2 style="color:#58a6ff;">Ready to Analyse</h2>
                <p style="color:#8b949e;font-size:15px;">
                    Select a stock in the sidebar and click <strong>Analyse Stock</strong>
                    to generate AI-powered trading signals.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        result = st.session_state.signal_result
        risk = st.session_state.risk
        df = st.session_state.df
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        sig_color = SIGNAL_COLORS.get(result.signal, "#9e9e9e")
        sig_emoji = SIGNAL_EMOJI.get(result.signal, "📊")
        price_change = pct_change(float(prev["Close"]), float(last["Close"]))

        comp_display = st.session_state.get("company_name", result.symbol)
        # ── Top Signal Card ─────────────────────────────────────────────────
        # ── Top Signal Card ─────────────────────────────────────────────────
        change_bg = 'rgba(0,230,118,0.15)' if price_change >= 0 else 'rgba(255,23,68,0.15)'
        change_fg = '#00e676' if price_change >= 0 else '#ff1744'
        change_border = 'rgba(0,230,118,0.3)' if price_change >= 0 else 'rgba(255,23,68,0.3)'
        arrow = '▲' if price_change >= 0 else '▼'

        st.markdown(
            f"""<div class="hero-signal-card" style="border: 1px solid {sig_color}55; box-shadow: 0 20px 60px -10px {sig_color}25;">
<div style="display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,0.05); padding:6px 16px; border-radius:20px; border:1px solid rgba(255,255,255,0.1); margin-bottom:12px; flex-wrap:wrap; justify-content:center;">
<span style="font-size:18px;">{sig_emoji}</span>
<span class="mono-font" style="font-size:13px; font-weight:700; color:#8b949e; letter-spacing:0.08em; text-transform:uppercase;">{result.symbol} • EQUITIES</span>
<span class="mono-font" style="font-size:12px; font-weight:700; color:#58a6ff; background:rgba(88,166,255,0.12); padding:3px 10px; border-radius:12px; border:1px solid rgba(88,166,255,0.25);">⏳ Signal Active: {result.signal_age_days} Days</span>
</div>
<div style="font-size:24px; font-weight:800; color:#f0f6fc; margin-bottom:8px;">{comp_display}</div>
<div style="font-size:42px; font-weight:800; color:{sig_color}; text-shadow:0 0 35px {sig_color}88; margin-bottom:10px; letter-spacing:-0.02em;">{result.signal}</div>
<div class="mono-font" style="font-size:22px; color:#f0f6fc; font-weight:700; margin-bottom:20px;">₹{last['Close']:,.2f} <span style="font-size:15px; font-weight:600; padding:3px 10px; border-radius:8px; margin-left:8px; background:{change_bg}; color:{change_fg}; border:1px solid {change_border}">{arrow} {abs(price_change):.2f}%</span></div>
<div style="font-size:12px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">Signal Confidence Score</div>
<div style="background:rgba(0,0,0,0.4); border-radius:10px; height:10px; width:65%; margin:0 auto 14px; padding:2px; border:1px solid rgba(255,255,255,0.06);"><div style="background:linear-gradient(90deg, {sig_color}bb, {sig_color}); height:100%; border-radius:8px; width:{result.confidence}%; box-shadow:0 0 12px {sig_color}aa;"></div></div>
<div class="mono-font" style="font-size:30px; font-weight:800; color:{sig_color}; text-shadow:0 0 20px {sig_color}66;">{result.confidence:.1f}%</div>
</div>""",
            unsafe_allow_html=True,
        )


        # ── Key Metrics Row ──────────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

        metrics = [
            (m1, "Entry Price", f"₹{risk.entry_price:,.2f}", "#58a6ff"),
            (m2, "Stop Loss", f"₹{risk.stop_loss:,.2f}", "#ff1744"),
            (m3, "Take Profit", f"₹{risk.take_profit:,.2f}", "#00e676"),
            (m4, "Risk:Reward", f"1:{risk.risk_reward:.1f}", "#ffb300"),
            (m5, "Active Days", f"{result.signal_age_days} Days", "#00e5ff"),
            (m6, "Horizon", f"{result.recommended_horizon}", "#ba68c8"),
            (m7, "Max Position", f"{risk.max_position_size:,} shares", "#79c0ff"),
        ]

        for col, label, value, color in metrics:
            col.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        st.markdown("<br>", unsafe_allow_html=True)

        # ── MTF Alignment Matrix Card ──────────────────────────────────────
        if hasattr(result, "mtf_result") and result.mtf_result:
            mtf = result.mtf_result
            st.markdown('<div class="section-header">⏱️ Multi-Timeframe (MTF) Alignment Matrix</div>', unsafe_allow_html=True)
            tf_cols = st.columns(3)
            tf_data = [
                (tf_cols[0], "1W Macro Trend", mtf.trends.get("1W")),
                (tf_cols[1], "1D Setup Trend", mtf.trends.get("1D")),
                (tf_cols[2], "1H Micro Entry", mtf.trends.get("1H")),
            ]
            for col, title, item in tf_data:
                if item:
                    color = "#00e676" if item.trend == "Bullish" else "#ff1744" if item.trend == "Bearish" else "#8b949e"
                    col.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 3px solid {color}; text-align:center;">
                            <div class="metric-label">{title}</div>
                            <div class="mono-font" style="font-size:18px; font-weight:700; color:{color}; margin-top:4px;">
                                {item.trend.upper()}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f"""
                <div style="background:rgba(22,27,34,0.6); border:1px solid rgba(88,166,255,0.2); border-radius:12px; padding:12px 18px; margin-top:12px; font-size:13px; color:#c9d1d9; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div><strong>Confluence Status:</strong> <span style="color:#58a6ff; font-weight:700;">{mtf.alignment_status}</span></div>
                    <div style="color:#8b949e;">{mtf.description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Two columns: Indicator scores + Analysis ──────────────────────
        col_l, col_r = st.columns([1, 1.4])

        with col_l:
            st.markdown('<div class="section-header">📊 Indicator Breakdown</div>', unsafe_allow_html=True)

            # Indicator score bars
            indicator_names = {
                "ema": "EMA Trend",
                "rsi": "RSI Momentum",
                "macd": "MACD Signal",
                "supertrend": "Supertrend",
                "adx": "ADX Strength",
                "volume": "Volume",
                "vwap": "VWAP Position",
            }
            weights = {"ema": 20, "rsi": 15, "macd": 20, "supertrend": 20, "adx": 10, "volume": 10, "vwap": 5}

            for key, label in indicator_names.items():
                raw = result.indicator_scores.get(key, 0)
                weight = weights.get(key, 10)
                pct = (raw / weight * 100) if weight > 0 else 50
                pct = max(0, min(100, pct))
                bar_color = "#00e676" if pct >= 60 else "#ffd740" if pct >= 40 else "#f44336"

                st.markdown(
                    f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:12px;color:#8b949e;margin-bottom:4px;">
                            <span>{label}</span>
                            <span style="color:{bar_color}">{pct:.0f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;">
                            <div style="background:{bar_color};width:{pct}%;height:6px;border-radius:4px;
                                        transition:width 0.4s ease;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Last bar values
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📌 Last Bar Summary</div>', unsafe_allow_html=True)
            last_row = df.iloc[-1]
            summary_items = []
            for col_name, fmt in [
                ("RSI", "{:.1f}"),
                ("ADX", "{:.1f}"),
                ("EMA_20", "₹{:.2f}"),
                ("EMA_50", "₹{:.2f}"),
                ("ATR", "₹{:.2f}"),
                ("VWAP", "₹{:.2f}"),
            ]:
                if col_name in df.columns:
                    val = last_row[col_name]
                    if pd.notna(val):
                        summary_items.append((col_name.replace("_", " "), fmt.format(val)))

            for name, val in summary_items:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;
                                padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);
                                font-size:13px;">
                        <span style="color:#8b949e;">{name}</span>
                        <span style="color:#e6edf3;font-weight:500;">{val}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_r:
            st.markdown('<div class="section-header">🤖 AI Trade Explanation</div>', unsafe_allow_html=True)

            # Generate AI explanation (rule-based fallback + optional LLM)
            ai_text = _generate_ai_explanation(result, risk)
            st.markdown(
                f'<div class="ai-box">{ai_text}</div>',
                unsafe_allow_html=True,
            )

            # News Sentiment Section
            if hasattr(result, "news_result") and result.news_result:
                news = result.news_result
                s_color = "#00e676" if "Bullish" in news.sentiment_label else "#ff1744" if "Bearish" in news.sentiment_label else "#ffb300"
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-header">📰 News Sentiment Intelligence (<span style="color:{s_color}">{news.sentiment_label}</span>)</div>', unsafe_allow_html=True)
                for art in news.articles[:3]:
                    art_color = "#00e676" if art.sentiment_label == "Bullish" else "#ff1744" if art.sentiment_label == "Bearish" else "#8b949e"
                    st.markdown(
                        f"""
                        <div class="reason-card" style="border-left-color:{art_color}; font-size:12.5px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                                <a href="{art.link}" target="_blank" style="color:#e6edf3; text-decoration:none; font-weight:600;">{art.title[:75]}…</a>
                                <span style="color:{art_color}; font-weight:700; font-size:10px; padding:1px 6px; border-radius:6px; background:{art_color}22;">{art.sentiment_label}</span>
                            </div>
                            <div style="font-size:10px; color:#8b949e;">{art.published}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Signal Reasons</div>', unsafe_allow_html=True)
            for reason in result.reasons[:10]:
                st.markdown(
                    f'<div class="reason-card">• {reason}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk Card & Monte Carlo Simulation ───────────────────────────────
        st.markdown('<div class="section-header">🛡️ Risk & Monte Carlo Quantitative Simulation (1,000 Iterations)</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        risk_metrics = [
            (rc1, "Capital Allocation", format_inr(risk.capital_allocation)),
            (rc2, "Risk Amount", format_inr(risk.risk_amount)),
            (rc3, "Stop Distance", f"{risk.stop_pct:.2f}%"),
            (rc4, "ATR Stop", format_inr(risk.atr_stop)),
        ]
        for col, label, val in risk_metrics:
            col.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:18px;">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if hasattr(result, "mc_result") and result.mc_result:
            mc = result.mc_result
            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc_color = "#00e676" if mc.win_probability >= 55 else "#ff1744" if mc.win_probability <= 45 else "#ffb300"
            mc_items = [
                (mc1, "Win Probability (PoP)", f"{mc.win_probability:.1f}%", mc_color),
                (mc2, "Expected Value (EV)", f"₹{mc.expected_value:,.2f}", "#58a6ff"),
                (mc3, "Value at Risk (95%)", f"₹{mc.var_95:,.2f}", "#ff1744"),
                (mc4, "20-Day Target (P50)", f"₹{mc.median_price:,.2f}", "#00e5ff"),
            ]
            for col, label, val, c in mc_items:
                col.markdown(
                    f"""
                    <div class="metric-card" style="border-top: 3px solid {c};">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="font-size:18px; color:{c};">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)


        # ── Better Stock Alternatives ──────────────────────────────────────────
        if "better_alternatives" in st.session_state and st.session_state.better_alternatives:
            sec_title = st.session_state.get("sector_category", "Sector")
            st.markdown(f'<div class="section-header">🌟 Better {sec_title} Alternatives (Industry Peers)</div>', unsafe_allow_html=True)
            alt_cols = st.columns(len(st.session_state.better_alternatives))

            for idx, (col, alt) in enumerate(zip(alt_cols, st.session_state.better_alternatives)):
                alt_sig_color = SIGNAL_COLORS.get(alt["signal"], "#58a6ff")
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 3px solid {alt_sig_color}; text-align:left; padding:16px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span class="mono-font" style="font-size:14px; font-weight:700; color:#f0f6fc;">{alt['symbol']}</span>
                                <span style="font-size:11px; font-weight:700; color:{alt_sig_color}; background:{alt_sig_color}22; padding:2px 8px; border-radius:10px; border:1px solid {alt_sig_color}44;">{alt['signal']}</span>
                            </div>
                            <div style="font-size:11px; color:#8b949e; margin-bottom:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{alt['name']}</div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:12px;">
                                <span style="color:#8b949e;">Confidence:</span>
                                <span class="mono-font" style="color:{alt_sig_color}; font-weight:700;">{alt['confidence']:.1f}%</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:12px;">
                                <span style="color:#8b949e;">Price:</span>
                                <span class="mono-font" style="color:#f0f6fc; font-weight:700;">₹{alt['price']:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:12px; font-size:12px;">
                                <span style="color:#8b949e;">Risk:Reward:</span>
                                <span class="mono-font" style="color:#00e676; font-weight:700;">1:{alt['rr']:.1f}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"⚡ Analyze {alt['symbol'].replace('.NS','')}", key=f"btn_alt_{alt['symbol']}_{idx}", use_container_width=True):
                        st.session_state.selected_symbol = alt["symbol"]
                        load_and_analyse(alt["symbol"])
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)


        # ── Watchlist + Alert buttons ────────────────────────────────────────
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            in_wl = is_in_watchlist(selected_symbol)
            wl_label = "⭐ Remove from Watchlist" if in_wl else "⭐ Add to Watchlist"
            if st.button(wl_label, use_container_width=True):
                if in_wl:
                    remove_from_watchlist(selected_symbol)
                    st.success(f"Removed {selected_symbol} from watchlist")
                else:
                    add_to_watchlist(selected_symbol, exchange=exchange)
                    st.success(f"Added {selected_symbol} to watchlist")
                st.rerun()

        with btn_c2:
            if st.button("📱 Send Telegram Alert", use_container_width=True):
                msg = format_signal_message(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                )
                ok = send_telegram_alert(msg, chat_id=tg_chat or None)
                st.success("Telegram sent!") if ok else st.error("Telegram not configured or failed.")

        with btn_c3:
            if st.button("📧 Send Email Alert", use_container_width=True):
                subj, body = format_signal_email(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                    result.reasons,
                )
                ok = send_email_alert(subj, body, receiver=email_to or None)
                st.success("Email sent!") if ok else st.error("Email not configured or failed.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHART
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chart:
    if st.session_state.df is None:
        st.info("📊 Load a stock from the sidebar to see the chart.")
    else:
        df = st.session_state.df
        result = st.session_state.signal_result

        # Company info header
        try:
            info = get_company_info(result.symbol)
            last_price = df["Close"].iloc[-1]
            prev_price = df["Close"].iloc[-2] if len(df) > 1 else last_price
            chg = pct_change(float(prev_price), float(last_price))

            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Company</div>
                    <div style="font-size:14px;font-weight:600;color:#e6edf3">{info.get('name', result.symbol)}</div>
                    <div style="font-size:11px;color:#8b949e;margin-top:4px;">{info.get('sector', '')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            ci2.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Last Price</div>
                    <div class="metric-value">₹{last_price:,.2f}</div>
                    <div class="metric-delta" style="color:{'#00e676' if chg >= 0 else '#f44336'}">
                        {'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            ci3.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">52-Week High</div>
                    <div class="metric-value">₹{info.get('52wHigh', 0):,.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            ci4.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">52-Week Low</div>
                    <div class="metric-value">₹{info.get('52wLow', 0):,.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        st.markdown("<br>", unsafe_allow_html=True)

        fig = build_price_chart(
            df,
            result.symbol,
            show_ema=show_ema,
            show_supertrend=show_supertrend,
            show_bollinger=show_bollinger,
            show_vwap=show_vwap,
            show_volume=show_volume,
            show_rsi=show_rsi,
            show_macd=show_macd,
            height=750,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Raw data expander
        with st.expander("📋 View Raw Data"):
            display_cols = [c for c in df.columns if c in [
                "Open", "High", "Low", "Close", "Volume",
                "EMA_20", "EMA_50", "RSI", "MACD", "ADX", "ATR",
                "Supertrend", "Supertrend_Direction",
            ]]
            st.dataframe(
                df[display_cols].tail(50).style.format("{:.2f}"),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.markdown("### ⚡ Strategy Backtesting")

    bt_col1, bt_col2 = st.columns([1, 2])

    with bt_col1:
        st.markdown("#### Configuration")
        bt_symbol = st.selectbox(
            "Stock",
            options=[s["symbol"] for s in NIFTY50_STOCKS],
            format_func=lambda s: f"{next((x['name'] for x in NIFTY50_STOCKS if x['symbol']==s), s)} ({s})",
            key="bt_symbol",
        )
        bt_period = st.selectbox("Period", ["1 Year", "2 Years", "3 Years", "5 Years"], key="bt_period")
        bt_period_map = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
        bt_capital = st.number_input(
            "Initial Capital (₹)", min_value=10_000, max_value=10_000_000,
            value=BT_DEFAULT_CAPITAL, step=10_000, format="%d", key="bt_capital",
        )
        run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    with bt_col2:
        if run_bt:
            with st.spinner("Running backtest…"):
                try:
                    bt_df_raw = fetch_ohlcv(bt_symbol, interval="1d", period=bt_period_map[bt_period])
                    bt_df = compute_all_indicators(bt_df_raw)
                    bt_result = run_backtest(bt_df, initial_capital=bt_capital)

                    # Metrics row
                    bm1, bm2, bm3 = st.columns(3)
                    bm4, bm5, bm6 = st.columns(3)

                    def _bt_metric(col, label, val, color=None):
                        style = f"color:{color};" if color else ""
                        col.markdown(
                            f"""<div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value" style="font-size:20px;{style}">{val}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    color_np = "#00e676" if bt_result.net_profit >= 0 else "#f44336"
                    _bt_metric(bm1, "Net Profit", format_inr(bt_result.net_profit), color_np)
                    _bt_metric(bm2, "CAGR", f"{bt_result.cagr:.1f}%", color_np)
                    _bt_metric(bm3, "Sharpe Ratio", f"{bt_result.sharpe_ratio:.2f}")
                    _bt_metric(bm4, "Max Drawdown", f"{bt_result.max_drawdown:.1f}%", "#f44336")
                    _bt_metric(bm5, "Win Rate", f"{bt_result.win_rate:.1f}%")
                    _bt_metric(bm6, "Total Trades", str(bt_result.total_trades))

                    st.markdown("<br>", unsafe_allow_html=True)

                    m7, m8, m9 = st.columns(3)
                    _bt_metric(m7, "Profit Factor", f"{bt_result.profit_factor:.2f}")
                    _bt_metric(m8, "Sortino Ratio", f"{bt_result.sortino_ratio:.2f}")
                    _bt_metric(m9, "Avg Trade", format_inr(bt_result.avg_trade))

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Equity curve
                    if not bt_result.equity_curve.empty:
                        st.markdown("#### 📈 Equity Curve")
                        eq_fig = build_equity_curve(bt_result.equity_curve, bt_capital)
                        st.plotly_chart(eq_fig, use_container_width=True)

                    # Trade log
                    if not bt_result.trade_log.empty:
                        with st.expander("📋 Trade Log"):
                            st.dataframe(bt_result.trade_log, use_container_width=True)

                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
        else:
            st.markdown(
                """
                <div style="text-align:center;padding:60px 20px;color:#8b949e;">
                    <div style="font-size:48px;margin-bottom:16px;">⚡</div>
                    <p>Configure parameters and click <strong>Run Backtest</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_scanner:
    st.markdown("### 🔍 Market Scanner")

    sc_col1, sc_col2 = st.columns([1, 3])

    with sc_col1:
        scan_index = st.selectbox("Index", list(INDEX_GROUPS.keys()))
        scan_filter = st.multiselect(
            "Filter by Signal",
            ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
            default=["Strong Buy", "Buy"],
        )
        scan_min_confidence = st.slider("Min Confidence %", 0, 100, 60)
        run_scan = st.button("🔍 Scan Now", type="primary", use_container_width=True)

    with sc_col2:
        if run_scan:
            scan_stocks = INDEX_GROUPS.get(scan_index, NIFTY50_STOCKS)
            results_list = []

            progress = st.progress(0, text="Scanning…")
            for i, stock in enumerate(scan_stocks):
                sym = stock["symbol"]
                progress.progress((i + 1) / len(scan_stocks), text=f"Scanning {sym}…")
                try:
                    raw = fetch_ohlcv(sym, interval="1d", period="6mo")
                    enriched = compute_all_indicators(raw)
                    sig_r = generate_signal(sym, enriched)
                    last = enriched.iloc[-1]
                    prev = enriched.iloc[-2] if len(enriched) > 1 else last
                    chg = pct_change(float(prev["Close"]), float(last["Close"]))

                    results_list.append({
                        "Symbol": sym,
                        "Name": stock["name"],
                        "Price": f"₹{last['Close']:,.2f}",
                        "Change%": f"{chg:+.2f}%",
                        "Signal": sig_r.signal,
                        "Confidence": f"{sig_r.confidence:.1f}%",
                        "RSI": f"{last.get('RSI', 0):.1f}" if pd.notna(last.get("RSI", None)) else "N/A",
                        "ADX": f"{last.get('ADX', 0):.1f}" if pd.notna(last.get("ADX", None)) else "N/A",
                    })
                except Exception:
                    pass

            progress.empty()

            if results_list:
                scan_df = pd.DataFrame(results_list)

                # Filter by signal type
                if scan_filter:
                    scan_df = scan_df[scan_df["Signal"].isin(scan_filter)]

                # Filter by confidence
                scan_df["_conf"] = scan_df["Confidence"].str.replace("%", "").astype(float)
                scan_df = scan_df[scan_df["_conf"] >= scan_min_confidence].drop("_conf", axis=1)

                if scan_df.empty:
                    st.warning("No stocks matched the current filter criteria.")
                else:
                    st.markdown(f"**Found {len(scan_df)} matches in {scan_index}**")

                    # Color-code signal column
                    def _style_signal(val):
                        color = SIGNAL_COLORS.get(val, "#9e9e9e")
                        return f"color: {color}; font-weight: bold"

                    styled = scan_df.style.applymap(_style_signal, subset=["Signal"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.warning("No results returned from scanner.")
        else:
            st.markdown(
                """
                <div style="text-align:center;padding:60px;color:#8b949e;">
                    <div style="font-size:48px;margin-bottom:16px;">🔍</div>
                    <p>Select an index and click <strong>Scan Now</strong> to find trading opportunities.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════

with tab_watchlist:
    st.markdown("### ⭐ Watchlist")

    wl_add_col, wl_table_col = st.columns([1, 3])

    with wl_add_col:
        st.markdown("#### Add Stock")
        wl_new = st.text_input("Symbol (e.g. WIPRO)", key="wl_new_symbol")
        wl_name = st.text_input("Name (optional)", key="wl_new_name")
        wl_exch = st.selectbox("Exchange", ["NSE", "BSE"], key="wl_exch")
        if st.button("➕ Add to Watchlist", use_container_width=True):
            if wl_new.strip():
                sym = normalise_symbol(wl_new.strip(), wl_exch)
                add_to_watchlist(sym, wl_name or sym, wl_exch)
                st.success(f"Added {sym}")
                st.rerun()
            else:
                st.warning("Enter a symbol first.")

    with wl_table_col:
        watchlist = get_watchlist()
        if not watchlist:
            st.markdown(
                """
                <div style="text-align:center;padding:60px;color:#8b949e;">
                    <div style="font-size:48px;margin-bottom:16px;">⭐</div>
                    <p>Your watchlist is empty. Add stocks using the panel on the left.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Live signals for watchlist
            wl_refresh = st.button("🔄 Refresh Signals", key="wl_refresh")
            if wl_refresh:
                with st.spinner("Fetching latest signals…"):
                    wl_data = []
                    for item in watchlist:
                        sym = item["symbol"]
                        try:
                            wl_df = fetch_ohlcv(sym, interval="1d", period="6mo")
                            wl_df = compute_all_indicators(wl_df)
                            sig_r = generate_signal(sym, wl_df)
                            last = wl_df.iloc[-1]
                            prev = wl_df.iloc[-2]
                            chg = pct_change(float(prev["Close"]), float(last["Close"]))
                            wl_data.append({
                                "Symbol": sym,
                                "Name": item["name"],
                                "Price": f"₹{last['Close']:,.2f}",
                                "Change %": f"{chg:+.2f}%",
                                "Signal": f"{SIGNAL_EMOJI.get(sig_r.signal, '')} {sig_r.signal}",
                                "Confidence": f"{sig_r.confidence:.1f}%",
                                "Added": item["added_at"][:10],
                            })
                        except Exception:
                            wl_data.append({"Symbol": sym, "Name": item["name"], "Signal": "Error"})

                    st.dataframe(pd.DataFrame(wl_data), use_container_width=True, hide_index=True)
            else:
                # Show basic list with remove buttons
                for item in watchlist:
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    col_a.markdown(f"**{item['symbol']}**")
                    col_b.markdown(f"<span style='color:#8b949e'>{item['name'] or '—'}</span>", unsafe_allow_html=True)
                    if col_c.button("🗑", key=f"rm_{item['symbol']}"):
                        remove_from_watchlist(item["symbol"])
                        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📜 Recent Signal History")
    recent = get_recent_signals(limit=20)
    if recent:
        recent_df = pd.DataFrame(recent)[
            ["symbol", "signal", "confidence", "entry_price", "risk_reward", "generated_at"]
        ]
        recent_df.columns = ["Symbol", "Signal", "Confidence %", "Entry ₹", "RR Ratio", "Generated At"]
        st.dataframe(recent_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_alerts:
    st.markdown("### 🔔 Alerts & Notifications")

    al1, al2 = st.columns(2)

    with al1:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#161b22,#1c2333);
                        border:1px solid rgba(88,166,255,0.2);border-radius:12px;padding:20px;">
                <h4 style="color:#29b6f6;margin:0 0 12px;">📱 Telegram Setup</h4>
                <ol style="color:#8b949e;font-size:13px;padding-left:16px;">
                    <li>Create a Telegram bot via <strong>@BotFather</strong></li>
                    <li>Copy the bot token</li>
                    <li>Start a chat with your bot and get the chat ID</li>
                    <li>Enter credentials in the sidebar → Alert Settings</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with al2:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#161b22,#1c2333);
                        border:1px solid rgba(88,166,255,0.2);border-radius:12px;padding:20px;">
                <h4 style="color:#ff9800;margin:0 0 12px;">📧 Email Setup</h4>
                <ol style="color:#8b949e;font-size:13px;padding-left:16px;">
                    <li>Use a Gmail account</li>
                    <li>Enable <strong>App Passwords</strong> in Google Account</li>
                    <li>Generate an app-specific password</li>
                    <li>Enter credentials in the sidebar → Alert Settings</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🧪 Test Alerts")
    if st.session_state.signal_result:
        result = st.session_state.signal_result
        risk = st.session_state.risk
        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button("Test Telegram", use_container_width=True):
                msg = format_signal_message(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                )
                ok = send_telegram_alert(msg, chat_id=tg_chat or None)
                st.success("✅ Sent!") if ok else st.error("❌ Failed (check credentials).")
        with tc2:
            if st.button("Test Email", use_container_width=True):
                subj, body = format_signal_email(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                    result.reasons,
                )
                ok = send_email_alert(subj, body, receiver=email_to or None)
                st.success("✅ Sent!") if ok else st.error("❌ Failed (check credentials).")
    else:
        st.info("Load a stock first to test alerts.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:rgba(22,27,34,0.8);border-radius:12px;padding:20px;">
            <h4 style="color:#8b949e;margin:0 0 8px;">🔮 Future Alert Triggers</h4>
            <ul style="color:#8b949e;font-size:13px;">
                <li>New Buy / Sell signal detected</li>
                <li>Target price hit</li>
                <li>Stop loss triggered</li>
                <li>RSI overbought / oversold</li>
                <li>Volume spike detected</li>
                <li>Supertrend flip</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div style="text-align:center;padding:40px 20px 20px;color:#484f58;font-size:12px;">
        <div style="margin-bottom:8px;">
            {APP_NAME} v{APP_VERSION} — Built with ❤️ using Streamlit & Python
        </div>
        <div>
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only.
            Not financial advice. Always do your own research.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Auto-refresh rerun
if auto_refresh:
    time.sleep(1)
    st.rerun()


