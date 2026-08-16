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

NIFTY_SMALLCAP: List[Dict[str, str]] = [
    {"symbol": "HAL.NS", "name": "Hindustan Aeronautics"},
    {"symbol": "BEL.NS", "name": "Bharat Electronics"},
    {"symbol": "IRFC.NS", "name": "Indian Railway Finance"},
    {"symbol": "RVNL.NS", "name": "Rail Vikas Nigam"},
    {"symbol": "BHEL.NS", "name": "Bharat Heavy Electricals"},
    {"symbol": "IOC.NS", "name": "Indian Oil Corporation"},
    {"symbol": "GAIL.NS", "name": "GAIL India"},
    {"symbol": "RECLTD.NS", "name": "REC Ltd"},

    {"symbol": "PFC.NS", "name": "Power Finance Corp"},
    {"symbol": "SUZLON.NS", "name": "Suzlon Energy"},
    {"symbol": "ZOMATO.NS", "name": "Eternal / Zomato"},
    {"symbol": "JIOFIN.NS", "name": "Jio Financial Services"},
    {"symbol": "POLYCAB.NS", "name": "Polycab India"},
    {"symbol": "TRENT.NS", "name": "Trent Ltd"},
    {"symbol": "IRCTC.NS", "name": "IRCTC"},
    {"symbol": "TATAPOWER.NS", "name": "Tata Power"},
    {"symbol": "NHPC.NS", "name": "NHPC Ltd"},
    {"symbol": "MAZDOCK.NS", "name": "Mazagon Dock Shipbuilders"},
    {"symbol": "BDL.NS", "name": "Bharat Dynamics"},
    {"symbol": "YESBANK.NS", "name": "Yes Bank"},
    {"symbol": "IDEA.NS", "name": "Vodafone Idea"},
    {"symbol": "DLF.NS", "name": "DLF Ltd"},
    {"symbol": "SAIL.NS", "name": "Steel Authority of India"},
    {"symbol": "NMDC.NS", "name": "NMDC Ltd"},
    {"symbol": "NATIONALUM.NS", "name": "National Aluminium"},
    {"symbol": "OIL.NS", "name": "Oil India"},
    {"symbol": "HUDCO.NS", "name": "HUDCO"},
    {"symbol": "IREDA.NS", "name": "IREDA"},
    {"symbol": "INDIHOTEL.NS", "name": "Indian Hotels Company"},
    {"symbol": "TATAELXSI.NS", "name": "Tata Elxsi"},
    {"symbol": "DIXON.NS", "name": "Dixon Technologies"},
    {"symbol": "KPITTECH.NS", "name": "KPIT Technologies"},
]

ALL_STOCKS: List[Dict[str, str]] = NIFTY50_STOCKS + NIFTY_MIDCAP + NIFTY_SMALLCAP

# ─────────────────────────────────────────────
# Sector & Category Pools for Peer Alternatives
# ─────────────────────────────────────────────
SECTOR_POOLS: Dict[str, List[Dict[str, str]]] = {

    "IT & Software": [
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
        {"symbol": "INFY.NS", "name": "Infosys"},
        {"symbol": "WIPRO.NS", "name": "Wipro"},
        {"symbol": "HCLTECH.NS", "name": "HCL Technologies"},
        {"symbol": "TECHM.NS", "name": "Tech Mahindra"},
        {"symbol": "LTIM.NS", "name": "LTIMindtree"},
        {"symbol": "PERSISTENT.NS", "name": "Persistent Systems"},
        {"symbol": "COFORGE.NS", "name": "Coforge"},
        {"symbol": "MPHASIS.NS", "name": "Mphasis"},
    ],
    "Banking & Financials": [
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
        {"symbol": "SBIN.NS", "name": "State Bank of India"},
        {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
        {"symbol": "AXISBANK.NS", "name": "Axis Bank"},
        {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
        {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv"},
        {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank"},
    ],
    "Automobiles": [
        {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
        {"symbol": "M&M.NS", "name": "Mahindra & Mahindra"},
        {"symbol": "MARUTI.NS", "name": "Maruti Suzuki"},
        {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp"},
        {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto"},
        {"symbol": "EICHERMOT.NS", "name": "Eicher Motors"},
    ],
    "Pharma & Healthcare": [
        {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical"},
        {"symbol": "CIPLA.NS", "name": "Cipla"},
        {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories"},
        {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories"},
        {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals"},
        {"symbol": "SYNGENE.NS", "name": "Syngene International"},
    ],
    "Energy & Utilities": [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
        {"symbol": "NTPC.NS", "name": "NTPC"},
        {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation"},
        {"symbol": "ONGC.NS", "name": "Oil & Natural Gas Corp"},
        {"symbol": "BPCL.NS", "name": "Bharat Petroleum"},
        {"symbol": "COALINDIA.NS", "name": "Coal India"},
    ],
    "Metals & Mining": [
        {"symbol": "TATASTEEL.NS", "name": "Tata Steel"},
        {"symbol": "HINDALCO.NS", "name": "Hindalco Industries"},
        {"symbol": "JSWSTEEL.NS", "name": "JSW Steel"},
        {"symbol": "COALINDIA.NS", "name": "Coal India"},
    ],
    "FMCG & Consumer": [
        {"symbol": "ITC.NS", "name": "ITC"},
        {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
        {"symbol": "NESTLEIND.NS", "name": "Nestle India"},
        {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products"},
        {"symbol": "BRITANNIA.NS", "name": "Britannia Industries"},
        {"symbol": "ASIANPAINT.NS", "name": "Asian Paints"},
        {"symbol": "TITAN.NS", "name": "Titan Company"},
    ],
    "Industrials & Cement": [
        {"symbol": "LT.NS", "name": "Larsen & Toubro"},
        {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement"},
        {"symbol": "GRASIM.NS", "name": "Grasim Industries"},
        {"symbol": "ADANIENT.NS", "name": "Adani Enterprises"},
        {"symbol": "ADANIPORTS.NS", "name": "Adani Ports"},
    ],
    "Defense & Capital Goods": [
        {"symbol": "HAL.NS", "name": "Hindustan Aeronautics"},
        {"symbol": "BEL.NS", "name": "Bharat Electronics"},
        {"symbol": "MAZDOCK.NS", "name": "Mazagon Dock Shipbuilders"},
        {"symbol": "BDL.NS", "name": "Bharat Dynamics"},
        {"symbol": "BHEL.NS", "name": "Bharat Heavy Electricals"},
    ],
}


INDEX_GROUPS: Dict[str, List[Dict[str, str]]] = {
    "ALL STOCKS (100+ Liquid NSE)": ALL_STOCKS,
    "NIFTY 50": NIFTY50_STOCKS,
    "NIFTY MIDCAP": NIFTY_MIDCAP,
    "NIFTY SMALLCAP & MOMENTUM": NIFTY_SMALLCAP,
    "IT & SOFTWARE": SECTOR_POOLS["IT & Software"],
    "BANKING & FINANCIALS": SECTOR_POOLS["Banking & Financials"],
    "AUTOMOBILES": SECTOR_POOLS["Automobiles"],
    "PHARMA & HEALTHCARE": SECTOR_POOLS["Pharma & Healthcare"],
    "ENERGY & UTILITIES": SECTOR_POOLS["Energy & Utilities"],
    "METALS & MINING": SECTOR_POOLS["Metals & Mining"],
    "FMCG & CONSUMER": SECTOR_POOLS["FMCG & Consumer"],
    "INDUSTRIALS & CEMENT": SECTOR_POOLS["Industrials & Cement"],
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
    "Strong Buy": "▲▲",
    "Buy": "▲",
    "Hold": "●",
    "Sell": "▼",
    "Strong Sell": "▼▼",
}

