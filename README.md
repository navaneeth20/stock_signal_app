---
title: StockSense AI Signal App
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 📈 StockSense AI — AI-Powered Indian Stock Market Signal Dashboard

> **Real-time trading signals powered by technical analysis and AI for Indian equities (NSE/BSE)**

---

## 🚀 Features

- Live NSE/BSE data via Yahoo Finance with caching
- 8+ Technical Indicators: EMA, RSI, MACD, Supertrend, ADX, ATR, Bollinger Bands, VWAP
- Multi-confirmation Signal Engine (Strong Buy → Strong Sell)
- Weighted Confidence Scoring (0–100%)
- AI-generated Trade Explanation (OpenAI / rule-based fallback)
- Interactive Plotly Candlestick Charts
- Risk Management: Entry / SL / TP / RRR / Position Sizing
- Vectorised Backtesting (CAGR, Sharpe, Sortino, Drawdown, Win Rate)
- Market Scanner (NIFTY 50, Midcap)
- Watchlist with SQLite persistence
- Telegram & Email Alerts
- Premium Dark Mode UI

---

## 📁 Project Structure

```
stock_signal_app/
├── app.py                      # Main Streamlit UI
├── config.py                   # All constants and configuration
├── requirements.txt
├── README.md
├── data/
│   ├── fetch_data.py           # yfinance data fetcher with retry + cache
│   └── cache.py                # 2-level cache (memory + SQLite)
├── indicators/
│   ├── ema.py                  # EMA 20/50/200
│   ├── rsi.py                  # RSI 14
│   ├── macd.py                 # MACD + Signal + Histogram
│   ├── supertrend.py           # Supertrend (ATR-based)
│   ├── adx.py                  # ADX + DI+/DI-
│   ├── atr.py                  # ATR
│   ├── bollinger.py            # Bollinger Bands
│   └── vwap.py                 # VWAP (rolling)
├── strategies/
│   ├── signal_engine.py        # Multi-confirmation signal generator
│   ├── scoring.py              # Weighted confidence scoring
│   └── risk.py                 # ATR stops + position sizing
├── backtesting/
│   └── backtest.py             # Vectorised backtester
├── charts/
│   └── candlestick.py          # Plotly chart builder
├── alerts/
│   ├── telegram.py             # Telegram Bot alerts
│   └── email.py                # SMTP email alerts
├── database/
│   └── database.py             # SQLite watchlist + signal history
└── utils/
    └── helpers.py              # Formatting + date utilities
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure AI and Alerts

Create a `.env` file in the project root:

```env
# OpenAI (for AI explanations — optional)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4o-mini

# Telegram alerts — optional
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email alerts — optional
EMAIL_SENDER=your@gmail.com
EMAIL_PASSWORD=app_specific_password
EMAIL_RECEIVER=recipient@email.com
```

### 3. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Signal Confidence Weights

| Indicator | Weight |
|---|---|
| EMA | 20% |
| MACD | 20% |
| Supertrend | 20% |
| RSI | 15% |
| ADX | 10% |
| Volume | 10% |
| VWAP | 5% |

---

## Risk Management

- Stop Loss: 1.5x ATR below entry
- Take Profit: 3.0x ATR above entry
- Position Size: Fixed fractional (2% risk per trade by default)

---

## Future Roadmap

- Kite Connect / AngelOne / Upstox live broker integration
- Paper trading and live trading mode
- LSTM price prediction
- XGBoost signal classifier
- Portfolio optimisation
- News sentiment analysis
- Options chain analysis
- FII/DII flow data
- Sector rotation tracker
- Candlestick pattern detection
- Multi-timeframe confirmation
- AI trade journal

---

## Disclaimer

This tool is for educational and research purposes only.
It does not constitute financial advice. Always conduct your own research and consult a SEBI-registered financial advisor before making investment decisions.
