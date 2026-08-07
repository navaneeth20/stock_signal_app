# StockSense AI Pro Terminal — System Architecture & Reference Documentation

> **Version**: 3.2.0 (Enterprise Institutional Quantitative Terminal)  
> **Repository**: [github.com/navaneeth20/stock_signal_app](https://github.com/navaneeth20/stock_signal_app)  
> **Platform**: Streamlit / Python 3.10+ / SQLite3 / YFinance / Gemini AI / Plotly  

---

## 1. Executive Summary & Overview

**StockSense AI Pro Terminal** is an enterprise-grade quantitative market analysis, technical signal generation, and institutional equity research platform designed specifically for the Indian Equity Markets (NSE & BSE).

The application combines multi-timeframe technical indicator calculations, weighted signal confidence scoring, ATR-based risk management, Monte Carlo risk simulation, market-wide sector scanners, persistent user identity tracking, end-of-day search audit logging, and LLM-powered institutional equity research prompts (via Google Gemini & OpenAI).

---

## 2. System Architecture & Component Structure

```
stock_signal_app/
├── app.py                      # Main Streamlit UI, Terminal Layout & State Management
├── config.py                   # Central Configuration, Stock Lists, Weights & Ticker Mappings
├── requirements.txt            # Python Dependencies
├── README.md                   # Quick Start & Overview
├── DOCUMENTATION.md            # Comprehensive Reference Documentation
│
├── data/
│   ├── fetch_data.py           # Resilient YFinance Fetcher, Symbol Normaliser & Ticker Aliases
│   └── cache.py                # 2-Level Caching Engine (Memory + SQLite)
│
├── database/
│   ├── database.py             # SQLite Schema Definitions & Database Helper Functions
│   └── __init__.py             # Database Exports & Interface
│
├── indicators/
│   ├── ema.py                  # Exponential Moving Averages (20, 50, 200)
│   ├── rsi.py                  # Relative Strength Index (RSI-14)
│   ├── macd.py                 # Moving Average Convergence Divergence
│   ├── supertrend.py           # Supertrend (ATR Multiplier)
│   ├── adx.py                  # Average Directional Index & DI+/DI-
│   ├── atr.py                  # Average True Range
│   ├── bollinger.py            # Bollinger Bands (Upper, Middle, Lower)
│   └── vwap.py                 # Volume Weighted Average Price
│
├── strategies/
│   ├── signal_engine.py        # Multi-Confirmation Signal Engine & MTF Matrix
│   ├── scoring.py              # Weighted Confidence Scoring (0-100%)
│   └── risk.py                 # ATR Stop Loss, Take Profit, Position Sizing & Monte Carlo
│
├── backtesting/
│   └── backtest.py             # Vectorised Historical Backtesting Engine
│
├── charts/
│   └── candlestick.py          # Plotly Interactive Candlestick & Volume Charts
│
├── reports/
│   └── institutional_llm.py    # 13 Institutional Research Prompt Suite & Fallback Engine
│
├── alerts/
│   ├── telegram.py             # Telegram Bot Notification Engine
│   └── email.py                # SMTP Email Alert Dispatcher
│
├── tests/
│   ├── run_tests.py            # Test Runner & Validation Diagnostics
│   ├── test_indicators.py      # Indicator Calculation Tests
│   └── test_signals.py         # Signal Generation Verification
│
└── utils/
    ├── helpers.py              # INR Formatting, Percentage Change & Utility Functions
    └── quant_risk.py           # Monte Carlo Simulation Engine
```

---

## 3. Core Features & Functional Modules

### Module 1: User Identity & Authentication Gate
- **Database Table**: `users` (`id`, `name`, `phone`, `email`, `created_at`, `last_login`).
- **Features**:
  - **Quick Sign-In**: Profile dropdown selector restoring saved trader credentials in 1 click.
  - **Registration Form**: First-time user onboarding asking for Name, Phone (+91), and Email.
  - **Lookup Engine**: Allows lookup via email address or phone number.
  - **Sidebar Active Trader Badge**: Displays user initials avatar block (`[NK] Navaneeth Kumar`), verified credentials, and account logout/switch button.

### Module 2: Resilient Ticker Normalisation & Data Ingestion
- **File**: `data/fetch_data.py`
- **Symbol Normaliser**: Automatically handles space-stripped symbols, uppercase conversion, and appends `.NS` for National Stock Exchange tickers.
- **Symbol Alias Engine**: Resolves irregular exchange ticker symbols.
  - *Example*: Maps `ION EXCHANGE`, `IONEXCHANGE`, and `ION EXCHANGE.NS` to Yahoo Finance's exact listed symbol `IONEXCHANG.NS`.
- **Retry Mechanism**: Implements exponential backoff (up to 3 retries) with yfinance ticker validation.

### Module 3: Signal Generation Engine & Indicator Scoring
- **File**: `strategies/signal_engine.py` & `strategies/scoring.py`
- **Indicators Computed**:
  - EMA 20, 50, 200
  - RSI (14)
  - MACD (12, 26, 9)
  - Supertrend (7, 3.0)
  - ADX (14)
  - Bollinger Bands (20, 2.0)
  - VWAP & Stochastic Oscillator
- **Weighted Signal Confidence Weights**:

| Indicator | Weight (%) | Criteria |
|---|---|---|
| EMA Alignment | 20% | Price > EMA20 > EMA50 > EMA200 |
| Supertrend | 20% | Supertrend Bullish / Bearish Flip |
| MACD | 20% | MACD Line > Signal Line & Histogram > 0 |
| RSI | 15% | 40 < RSI < 70 (Bullish Momentum zone) |
| ADX | 10% | ADX > 25 (Strong Trend Strength) |
| Volume Spike | 10% | Volume > 1.5x 20-period Moving Average Volume |
| VWAP | 5% | Close > VWAP |

- **Final Signal Classification**:
  - **STRONG BUY**: Score $\ge 75\%$
  - **BUY**: Score $\ge 60\%$
  - **HOLD**: Score between $40\%$ and $59\%$
  - **SELL**: Score $\le 39\%$
  - **STRONG SELL**: Score $\le 25\%$

### Module 4: Multi-Timeframe (MTF) Alignment Matrix
- Calculates indicator alignment across three distinct timeframes:
  1. **1W (Weekly)**: Macro Trend Direction
  2. **1D (Daily)**: Primary Setup Signal
  3. **1H (Hourly)**: Micro Entry Trigger
- Confirms whether higher timeframe trends support lower timeframe entries to prevent false breakouts.

### Module 5: Risk Management & Monte Carlo Simulation
- **File**: `strategies/risk.py` & `utils/quant_risk.py`
- **Risk Metrics**:
  - **Stop Loss**: $1.5 \times \text{ATR}$ below entry price.
  - **Take Profit**: $3.0 \times \text{ATR}$ above entry price.
  - **Risk:Reward Ratio**: Target fixed at $\ge 1:2.0$.
  - **Position Sizing**: Calculated dynamically based on Portfolio Capital ($\text{₹}$) and Max Risk Per Trade ($\%$).
- **Monte Carlo Risk Engine**: Runs 1,000 statistical trade trajectory simulations over 252 trading days to calculate expected drawdown, win rate distributions, and equity curves.

### Module 6: Market Scanner & Sector Breadth
- **Features**: Scans entire NIFTY 50, NIFTY NEXT 50, NIFTY MIDCAP 100, or custom 100+ liquid NSE stock universe.
- **Output Statistics**: Market advance/decline ratio, sector top gainers, high-confidence signals ($\ge 55\%$), and 1-click single-stock analysis triggers.

### Module 7: Watchlist & End-of-Day (EOD) Search Audit Log
- **Database Tables**: `watchlist`, `search_history`
- **Audit Engine**: Automatically records every search query performed by traders along with timestamp, user ID, stock name, signal, confidence, entry price, and source.
- **EOD Export**: 1-click CSV download of all search logs recorded throughout the trading day.

### Module 8: Institutional Equity Research Engine (LLM Prompt Suite)
- **File**: `reports/institutional_llm.py`
- **Prompt Suite (Gemini / OpenAI)**: Includes 13 institutional prompt templates based on BSE/NSE filings, annual reports, and concall transcripts:
  1. Full Business Model Breakdown
  2. Deep 5-Year Financial History Table
  3. Competitive Moat Analysis (Scored out of 10)
  4. Capital Allocation & Management Track Record
  5. Valuation Framework (DCF & Relative Valuation)
  6. Forensic Accounting & Red Flag Audit
  7. Forward Growth Drivers & 3-Year Catalysts
  8. Risk Matrix & Bear Case Scenario
  9. Key Ratios Deep-Dive
  10. Institutional & Promoter Holding Patterns
  11. Industry Dynamics & Market Share Trends
  12. Quarterly Earnings Performance Review
  13. Executive Investment Recommendation Summary

---

## 4. Database Schema Reference

The SQLite database file is located at `database/stock_app.db`.

```sql
-- User Identity Table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search History & EOD Audit Log Table
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    symbol TEXT NOT NULL,
    name TEXT,
    signal TEXT,
    confidence REAL,
    price REAL,
    source TEXT DEFAULT 'Search',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- Watchlist Table
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    symbol TEXT NOT NULL,
    name TEXT,
    exchange TEXT DEFAULT 'NSE',
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- Price & Signal Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    symbol TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    target_value REAL,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
```

---

## 5. Local Setup & Execution Guide

### Prerequisites
- Python 3.10 or higher
- Git

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/navaneeth20/stock_signal_app.git
   cd stock_signal_app
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables (Optional)**:
   Create a `.env` file in the root directory:
   ```env
   # Google Gemini API Key (For Institutional Research Prompts)
   GEMINI_API_KEY=your_gemini_api_key
   
   # OpenAI API Key (Optional fallback)
   OPENAI_API_KEY=your_openai_api_key
   
   # Telegram Notifications (Optional)
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

4. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

5. **Run Unit Tests & Diagnostic Suite**:
   ```bash
   python tests/run_tests.py
   ```

---

## 6. Enterprise UI Design System Guidelines

- **Typography**: `Plus Jakarta Sans`, `Inter`, and `JetBrains Mono` for ticker prices and quantitative metrics.
- **Theme Palette**: Deep Dark Slate (`#090D14`) with glassmorphism panels (`rgba(17, 24, 39, 0.75)`).
- **Navigation Tabs**: Clean uppercase text badges (`SIGNAL TERMINAL`, `TECHNICAL ANALYSIS`, `QUANT BACKTEST`, `MARKET SCANNER`, `WATCHLIST & AUDIT LOGS`, `INSTITUTIONAL RESEARCH`, `ALERT MANAGER`) with Streamlit default red/pink highlight lines disabled.
- **Icon Policy**: Pure vector SVGs and clean typography badges—zero informal cartoon emojis.

---

## 7. Disclaimer

*This software is created strictly for quantitative research, backtesting, and educational purposes. It does not constitute financial advice or SEBI-registered investment recommendations. Always conduct independent fundamental research and consult a certified financial advisor before trading real capital in live financial markets.*
