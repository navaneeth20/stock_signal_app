"""
backtesting/backtest.py
=======================
Pure-Python vectorised backtester.
Uses precomputed signals to simulate trade execution and compute metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import BT_COMMISSION, BT_DEFAULT_CAPITAL, RISK_FREE_RATE
from strategies.signal_engine import compute_all_indicators, generate_signal
from strategies.scoring import label_from_score

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Stores all backtesting metrics and trade history."""

    net_profit: float
    net_profit_pct: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade: float
    expectancy: float
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = BT_DEFAULT_CAPITAL


def _generate_signals_series(df: pd.DataFrame) -> pd.Series:
    """
    Vectorised signal generation for the full price series.
    Returns a Series of 1 (Buy) / -1 (Sell) / 0 (Hold).
    """
    signals = pd.Series(0, index=df.index)

    # Use rolling 60-bar windows to generate signals at each point
    for i in range(60, len(df)):
        window = df.iloc[: i + 1]
        try:
            # Quick scoring without re-running heavy compute
            from indicators.ema import ema_signal
            from indicators.macd import macd_signal
            from indicators.rsi import rsi_signal
            from indicators.supertrend import supertrend_signal
            from strategies.scoring import compute_score

            indicator_signals = {
                "ema": ema_signal(window),
                "rsi": rsi_signal(window),
                "macd": macd_signal(window),
                "supertrend": supertrend_signal(window),
                "adx": {"signal": 0, "score": 0, "reasons": []},
                "volume": {"signal": 0, "score": 0, "reasons": []},
                "vwap": {"signal": 0, "score": 0, "reasons": []},
            }
            score = compute_score(indicator_signals)["confidence"]
            label = label_from_score(score)
            if label in ("Strong Buy", "Buy"):
                signals.iloc[i] = 1
            elif label in ("Strong Sell", "Sell"):
                signals.iloc[i] = -1
        except Exception:
            pass

    return signals


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = BT_DEFAULT_CAPITAL,
    commission: float = BT_COMMISSION,
) -> BacktestResult:
    """
    Run a vectorised backtest on enriched OHLCV data.

    Args:
        df:              Enriched OHLCV DataFrame (must have all indicator columns).
        initial_capital: Starting capital in ₹.
        commission:      Round-trip commission fraction (e.g. 0.001 = 0.1%).

    Returns:
        BacktestResult with all performance metrics.
    """
    logger.info("Running backtest on %d bars…", len(df))

    signals = _generate_signals_series(df)

    # Simulate trades
    equity = initial_capital
    position = 0        # shares held
    entry_price = 0.0
    entry_date = df.index[0]
    trades: list[dict] = []
    equity_curve = []

    for i in range(len(df)):
        close = df["Close"].iloc[i]
        sig = signals.iloc[i]
        date = df.index[i]

        # Entry
        if sig == 1 and position == 0:
            shares = int(equity / (close * (1 + commission)))
            if shares > 0:
                cost = shares * close * (1 + commission)
                equity -= cost
                position = shares
                entry_price = close
                entry_date = date

        # Exit
        elif sig == -1 and position > 0:
            proceeds = position * close * (1 - commission)
            pnl = proceeds - (position * entry_price)
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": close,
                "shares": position,
                "pnl": round(pnl, 2),
                "return_pct": round((close / entry_price - 1) * 100, 2),
            })
            equity += proceeds
            position = 0

        # Mark-to-market equity
        mtm = equity + (position * close if position > 0 else 0)
        equity_curve.append({"date": date, "equity": mtm})

    # Close any open position at last price
    if position > 0:
        last_close = df["Close"].iloc[-1]
        proceeds = position * last_close * (1 - commission)
        pnl = proceeds - (position * entry_price)
        trades.append({
            "entry_date": entry_date,
            "exit_date": df.index[-1],
            "entry_price": entry_price,
            "exit_price": last_close,
            "shares": position,
            "pnl": round(pnl, 2),
            "return_pct": round((last_close / entry_price - 1) * 100, 2),
        })
        equity += proceeds

    # ── Metrics ──────────────────────────────────────────────────────────────
    eq_series = pd.Series(
        [e["equity"] for e in equity_curve],
        index=[e["date"] for e in equity_curve],
    )

    net_profit = equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    # CAGR
    n_days = max((df.index[-1] - df.index[0]).days, 1)
    years = n_days / 365.25
    if equity > 0 and years > 0:
        cagr = ((equity / initial_capital) ** (1.0 / years) - 1) * 100
    else:
        cagr = -100.0 if equity <= 0 else 0.0

    # Daily returns
    daily_ret = eq_series.pct_change().dropna()

    # Sharpe
    excess = daily_ret - (RISK_FREE_RATE / 252)
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if (not excess.empty and excess.std() > 1e-9) else 0.0

    # Sortino
    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std() * np.sqrt(252)) if (not downside.empty and downside.std() > 1e-9) else 0.0

    # Max Drawdown
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    # Trade stats
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        wins = trade_df[trade_df["pnl"] > 0]
        losses = trade_df[trade_df["pnl"] <= 0]
        win_rate = len(wins) / len(trade_df) * 100
        gross_profit = wins["pnl"].sum() if not wins.empty else 0
        gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 1e-9
        profit_factor = gross_profit / gross_loss
        avg_trade = trade_df["pnl"].mean()
        expectancy = (
            (win_rate / 100) * wins["pnl"].mean()
            - (1 - win_rate / 100) * abs(losses["pnl"].mean() if not losses.empty else 0)
        )
    else:
        win_rate = profit_factor = avg_trade = expectancy = 0

    return BacktestResult(
        net_profit=round(net_profit, 2),
        net_profit_pct=round(net_profit_pct, 2),
        cagr=round(cagr, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        max_drawdown=round(max_dd, 2),
        win_rate=round(win_rate, 2),
        profit_factor=round(profit_factor, 2),
        total_trades=len(trades),
        avg_trade=round(avg_trade, 2),
        expectancy=round(expectancy, 2),
        equity_curve=eq_series,
        trade_log=trade_df if not trade_df.empty else pd.DataFrame(),
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        initial_capital=initial_capital,
    )
