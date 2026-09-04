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

- NSE/BSE price data via Yahoo Finance, with a two-level cache
- 8+ Technical Indicators: EMA, RSI, MACD, Supertrend, ADX, ATR, Bollinger Bands, VWAP
- Multi-confirmation Signal Engine (Strong Buy → Strong Sell)
- Weighted Confidence Scoring (0–100%)
- AI-generated Trade Explanation (OpenAI / rule-based fallback)
- Interactive Plotly Candlestick Charts
- Risk Management: Entry / SL / TP / RRR / Position Sizing
- Backtesting with a buy-and-hold benchmark (CAGR, Sharpe, Sortino, Drawdown, Win Rate)
- Monte Carlo probability of profit, for long and short setups
- Market Scanner (NIFTY 50, Midcap, Smallcap, sector groups) with concurrent fetching
- Per-user Watchlist with SQLite persistence and password sign-in
- Telegram & Email Alerts
- Premium Dark Mode UI

### Where the data comes from

Everything priced in this app comes from **Yahoo Finance** (`yfinance`). There
is no direct NSE or BSE feed — "NSE" in the UI refers to the exchange the
`.NS` tickers trade on, not to a data source.

That matters for one thing in particular: **the app does not have FII, DII or
mutual-fund shareholding data**, because Yahoo does not carry it for Indian
listings. What the Signal Terminal shows instead is an *accumulation
footprint* derived from price and volume (up-day volume share, OBV slope,
close location value), clearly labelled as a proxy. Real institutional
holdings come from BSE/NSE quarterly shareholding filings and AMFI monthly
disclosures; wiring those in is on the roadmap.

Likewise, the Institutional Research tab sends prompts to an LLM that has **no
retrieval and no access to filings**. It is useful for qualitative framing —
business model, risk taxonomy, bull/bear cases — and unreliable for any
specific number. Every figure it produces is marked unverified.

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
│   └── backtest.py             # Event-driven backtester + benchmark
├── charts/
│   └── candlestick.py          # Plotly chart builder
├── alerts/
│   ├── telegram.py             # Telegram Bot alerts
│   └── email.py                # SMTP email alerts
├── database/
│   └── database.py             # SQLite users, watchlist, signal history
├── reports/
│   └── institutional_llm.py    # LLM research prompts + grounding guard
├── tests/
│   ├── test_indicators.py
│   ├── test_signals.py
│   ├── test_auth.py
│   └── test_data_integrity.py
└── utils/
    ├── helpers.py              # Formatting + date utilities
    └── quant_risk.py           # Monte Carlo / VaR
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

- Stop Loss: 1.5x ATR (below entry for longs, above for shorts)
- Take Profit: 2:1 reward-to-risk against that stop
- Position Size: Fixed fractional (2% risk per trade by default)

All levels come from `strategies/risk.py`, which is the single source of truth
for entry, stop and target.

---

## How the Backtest Works

Worth reading before trusting a number from it:

- It runs the **same** scoring path as the live signal engine — every
  indicator at its configured weight.
- A signal derived from bar *i*'s close is filled at bar *i+1*'s **open**.
- The ATR stop and target are applied, checked intrabar against High/Low. When
  one bar spans both levels, the stop is assumed to fill first.
- Costs are 0.2% per side plus 0.05% slippage — roughly brokerage + STT +
  exchange charges + GST + stamp duty for NSE delivery.
- **Buy-and-hold over the identical window is reported alongside.** A 24% CAGR
  means nothing if the stock itself did 31%.

---

## Running the Tests

```bash
pytest tests/ -v
```

### Checking the ticker list

Indian symbols change — companies rename (Zomato → Eternal), demerge
(Tata Motors → TMPV), and typos hide for a long time because a dead symbol only
produces a log warning and silently vanishes from every scan:

```bash
python tools/validate_tickers.py
```

Exits non-zero if anything in `config.py` no longer resolves. When replacing a
symbol, confirm it is genuinely the same company — one that merely resolves is
not good enough (`LTTS` is L&T Technology Services, not LTIMindtree).

### Pinned dependencies

`requirements.txt` carries tested bounds; `requirements.lock.txt` is an exact
freeze of a known-good environment:

```bash
pip install -r requirements.lock.txt
```

Streamlit is capped below 2.0 on purpose — the CSS in `app.py` targets
`data-testid` attributes (`stSidebar`, `stRadio`, `stFormSubmitButton`,
`stButton`) that Streamlit renames between releases.

---

## Future Roadmap

- Kite Connect / AngelOne / Upstox live broker integration
- Paper trading and live trading mode
- LSTM price prediction
- XGBoost signal classifier
- Portfolio optimisation
- News sentiment analysis
- Options chain analysis
- **Real FII/DII flow data** — BSE/NSE quarterly shareholding filings and AMFI
  monthly disclosures, to replace the current price/volume proxy
- **Grounded research** — feed real financials into the LLM prompts instead of
  asking the model to recall them
- Sector rotation tracker
- Candlestick pattern detection
- Multi-timeframe confirmation
- AI trade journal

---

## Disclaimer

This tool is for educational and research purposes only.
It does not constitute financial advice. Always conduct your own research and consult a SEBI-registered financial advisor before making investment decisions.
