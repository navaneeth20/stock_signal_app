"""
config.py
=========
Central configuration for the AI-Powered Indian Stock Signal App.
All tuneable constants, API keys, and defaults are defined here.
"""

import os
from pathlib import Path
from typing import Dict, List

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "database" / "signals.db"
ASSETS_DIR = BASE_DIR / "assets"

# ─────────────────────────────────────────────
# App Meta
# ─────────────────────────────────────────────
APP_NAME = "StockSense AI"
APP_VERSION = "1.0.0"
APP_TAGLINE = "AI-Powered Indian Stock Market Signals"

# ─────────────────────────────────────────────
# Market Data
# ─────────────────────────────────────────────
DEFAULT_EXCHANGE = "NSE"
DEFAULT_INTERVAL = "1d"
DEFAULT_PERIOD = "1y"
CACHE_TTL_SECONDS = 300  # 5 minutes

SUPPORTED_INTERVALS = {
    "1 Day": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo",
    "1 Hour": "1h",
    "15 Min": "15m",
    "5 Min": "5m",
}

# ─────────────────────────────────────────────
# Popular Indian Stocks (Yahoo Finance symbols)
# ─────────────────────────────────────────────
NIFTY50_STOCKS: List[Dict[str, str]] = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
    {"symbol": "INFY.NS", "name": "Infosys"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
    {"symbol": "SBIN.NS", "name": "State Bank of India"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank"},
    {"symbol": "WIPRO.NS", "name": "Wipro"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical"},
    {"symbol": "TITAN.NS", "name": "Titan Company"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki"},
    {"symbol": "NTPC.NS", "name": "NTPC"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation"},
    {"symbol": "ONGC.NS", "name": "Oil & Natural Gas Corp"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank"},
    {"symbol": "NESTLEIND.NS", "name": "Nestle India"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories"},
    {"symbol": "CIPLA.NS", "name": "Cipla"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries"},
    {"symbol": "SHREECEM.NS", "name": "Shree Cement"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra"},
    {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals"},
    {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises"},
    {"symbol": "COALINDIA.NS", "name": "Coal India"},
    {"symbol": "UPL.NS", "name": "UPL"},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
    {"symbol": "ITC.NS", "name": "ITC"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance"},
    {"symbol": "HDFCLIFE.NS", "name": "HDFC Life Insurance"},
]

NIFTY_MIDCAP: List[Dict[str, str]] = [
    {"symbol": "PERSISTENT.NS", "name": "Persistent Systems"},
    {"symbol": "COFORGE.NS", "name": "Coforge"},
    {"symbol": "MPHASIS.NS", "name": "Mphasis"},
    {"symbol": "LTIM.NS", "name": "LTIMindtree"},
    {"symbol": "PIIND.NS", "name": "PI Industries"},
    {"symbol": "SYNGENE.NS", "name": "Syngene International"},
    {"symbol": "CROMPTON.NS", "name": "Crompton Greaves Consumer"},
    {"symbol": "VOLTAS.NS", "name": "Voltas"},
    {"symbol": "CONCOR.NS", "name": "Container Corporation of India"},
]

INDEX_GROUPS: Dict[str, List[Dict[str, str]]] = {
    "NIFTY 50": NIFTY50_STOCKS,
    "NIFTY MIDCAP": NIFTY_MIDCAP,
}

# ─────────────────────────────────────────────
# Indicator Defaults
# ─────────────────────────────────────────────
EMA_SHORT = 20
EMA_MID = 50
EMA_LONG = 200
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
ADX_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
VWAP_PERIOD = 14
VOLUME_MA_PERIOD = 20

# ─────────────────────────────────────────────
# Signal Scoring Weights  (must sum to 100)
# ─────────────────────────────────────────────
SIGNAL_WEIGHTS: Dict[str, int] = {
    "ema": 20,
    "rsi": 15,
    "macd": 20,
    "supertrend": 20,
    "adx": 10,
    "volume": 10,
    "vwap": 5,
}

# ─────────────────────────────────────────────
# Signal Thresholds
# ─────────────────────────────────────────────
STRONG_BUY_THRESHOLD = 75
BUY_THRESHOLD = 55
SELL_THRESHOLD = 45
STRONG_SELL_THRESHOLD = 25

# ─────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────
DEFAULT_CAPITAL = 100_000        # ₹1 Lakh
DEFAULT_RISK_PER_TRADE = 0.02   # 2%
ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_RR = 2.0            # Default 2:1 RRR

# ─────────────────────────────────────────────
# Backtesting Defaults
# ─────────────────────────────────────────────
BT_DEFAULT_CAPITAL = 100_000
BT_COMMISSION = 0.001  # 0.1% round-trip per trade
RISK_FREE_RATE = 0.065  # 6.5% Indian 10Y bond

# ─────────────────────────────────────────────
# AI Explanation (OpenAI-compatible)
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
AI_MAX_TOKENS = 400

# ─────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
CHART_HEIGHT = 600
VOLUME_CHART_HEIGHT = 150
OSCILLATOR_HEIGHT = 200

SIGNAL_COLORS = {
    "Strong Buy": "#00e676",
    "Buy": "#69f0ae",
    "Hold": "#ffd740",
    "Sell": "#ff6e40",
    "Strong Sell": "#f44336",
}

SIGNAL_EMOJI = {
    "Strong Buy": "🚀",
    "Buy": "📈",
    "Hold": "⏸️",
    "Sell": "📉",
    "Strong Sell": "🔻",
}
