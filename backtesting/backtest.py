"""
backtesting/backtest.py
=======================
Event-driven backtester for the signal engine.

Design notes (these are the things that make a backtest honest):

  * It runs the **same** scoring path the live app uses — every indicator,
    with its configured weight. Pinning ADX/volume/VWAP to neutral, as the
    previous version did, measured a strategy the app does not trade.
  * Signals derived from bar *i*'s close are executed at bar *i+1*'s **open**.
    You cannot know a close-based signal and fill at that same close.
  * The ATR stop and target the UI displays are actually applied, checked
    intrabar against High/Low, and the stop is assumed to fill first when a
    single bar spans both levels (the conservative assumption).
  * Costs are charged per side and include slippage.
  * Buy-and-hold over the identical window is reported alongside, because a
    24% CAGR means nothing without knowing the stock did 31%.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BT_ATR_STOP_MULT,
    BT_ATR_TARGET_MULT,
    BT_COMMISSION,
    BT_DEFAULT_CAPITAL,
    BT_SLIPPAGE,
    BT_USE_STOPS,
    RISK_FREE_RATE,
)
from strategies.scoring import compute_score, label_from_score

logger = logging.getLogger(__name__)

# Bars of history the signal engine needs before it will produce a signal.
WARMUP_BARS = 60


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

    # Benchmark — buy and hold the same stock over the same window.
    benchmark_return_pct: float = 0.0
    benchmark_cagr: float = 0.0
    benchmark_max_drawdown: float = 0.0
    excess_return_pct: float = 0.0
    time_in_market_pct: float = 0.0
    exit_breakdown: dict = field(default_factory=dict)


def _generate_signals_series(df: pd.DataFrame) -> pd.Series:
    """
    Generate the label series the live engine would have produced at each bar.

    Uses the full indicator set with configured weights, evaluated on data
    available up to and including that bar. Every indicator column is causal
    (no centred or forward-looking windows), so slicing the precomputed frame
    is equivalent to recomputing on the window — and far cheaper.

    Returns:
        Series of 1 (Buy) / -1 (Sell) / 0 (Hold), indexed like ``df``.
    """
    from indicators.adx import adx_signal
    from indicators.ema import ema_signal
    from indicators.macd import macd_signal
    from indicators.rsi import rsi_signal
    from indicators.supertrend import supertrend_signal
    from indicators.vwap import vwap_signal
    from strategies.signal_engine import volume_signal

    signals = pd.Series(0, index=df.index, dtype=int)

    for i in range(WARMUP_BARS, len(df)):
        window = df.iloc[: i + 1]
        try:
            adx_s = adx_signal(window)
            indicator_signals = {
                "ema": ema_signal(window),
                "rsi": rsi_signal(window, adx_value=adx_s.get("adx_value")),
                "macd": macd_signal(window),
                "supertrend": supertrend_signal(window),
                "adx": adx_s,
                "volume": volume_signal(window),
                "vwap": vwap_signal(window),
            }
            confidence = compute_score(indicator_signals)["confidence"]
            label = label_from_score(confidence)
            if label in ("Strong Buy", "Buy"):
                signals.iloc[i] = 1
            elif label in ("Strong Sell", "Sell"):
                signals.iloc[i] = -1
        except Exception:  # noqa: BLE001 - one bad bar shouldn't kill the run
            logger.debug("Signal generation failed at bar %d", i, exc_info=True)

    return signals


def _annualised_stats(equity: pd.Series, periods_per_year: int = 252) -> tuple[float, float]:
    """
    Sharpe and Sortino for an equity curve.

    Sortino's denominator is downside *deviation* — the RMS of returns below
    the target, measured across all periods — not the standard deviation of
    only the negative subset. The subset's std is much smaller, which inflates
    the ratio by 2-3x.
    """
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0, 0.0

    excess = returns - (RISK_FREE_RATE / periods_per_year)
    mean_excess = excess.mean()

    std = excess.std()
    sharpe = (mean_excess / std * np.sqrt(periods_per_year)) if std > 1e-9 else 0.0

    downside_dev = np.sqrt((excess.clip(upper=0) ** 2).mean())
    sortino = (
        (mean_excess / downside_dev * np.sqrt(periods_per_year))
        if downside_dev > 1e-9
        else 0.0
    )

    return float(sharpe), float(sortino)


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative percentage."""
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max.replace(0, np.nan)
    result = drawdown.min()
    return float(result * 100) if pd.notna(result) else 0.0


def _cagr(start_value: float, end_value: float, years: float) -> float:
    """Compound annual growth rate as a percentage."""
    if years <= 0 or start_value <= 0:
        return 0.0
    if end_value <= 0:
        return -100.0
    return float(((end_value / start_value) ** (1.0 / years) - 1) * 100)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = BT_DEFAULT_CAPITAL,
    commission: float = BT_COMMISSION,
    slippage: float = BT_SLIPPAGE,
    use_stops: bool = BT_USE_STOPS,
    stop_mult: float = BT_ATR_STOP_MULT,
    target_mult: float = BT_ATR_TARGET_MULT,
) -> BacktestResult:
    """
    Run a long-only backtest on enriched OHLCV data.

    Args:
        df:              Enriched OHLCV DataFrame (all indicator columns present).
        initial_capital: Starting capital in ₹.
        commission:      Cost per side as a fraction (0.002 = 0.2%).
        slippage:        Additional fill slippage per side.
        use_stops:       Apply the ATR stop/target the UI displays.
        stop_mult:       ATR multiplier for the stop.
        target_mult:     ATR multiplier for the target.

    Returns:
        BacktestResult with strategy and benchmark metrics.
    """
    if len(df) <= WARMUP_BARS + 1:
        raise ValueError(
            f"Need more than {WARMUP_BARS + 1} bars to backtest, got {len(df)}"
        )

    logger.info("Running backtest on %d bars…", len(df))

    signals = _generate_signals_series(df)

    close = df["Close"].to_numpy(dtype=float)
    open_px = df["Open"].to_numpy(dtype=float) if "Open" in df.columns else close
    high = df["High"].to_numpy(dtype=float) if "High" in df.columns else close
    low = df["Low"].to_numpy(dtype=float) if "Low" in df.columns else close
    atr_arr = (
        df["ATR"].to_numpy(dtype=float)
        if "ATR" in df.columns
        else np.full(len(df), np.nan)
    )
    sig_arr = signals.to_numpy(dtype=int)

    cash = float(initial_capital)
    position = 0
    entry_price = 0.0
    entry_date = df.index[0]
    stop_price = 0.0
    target_price = 0.0

    trades: list[dict] = []
    equity_values: list[float] = []
    bars_in_market = 0

    def _record_exit(exit_px: float, exit_dt, reason: str) -> float:
        """Close the open position at exit_px and log the trade. Returns proceeds."""
        fill = exit_px * (1 - slippage)
        proceeds = position * fill * (1 - commission)
        cost_basis = position * entry_price * (1 + commission + slippage)
        pnl = proceeds - cost_basis
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_dt,
                "entry_price": round(entry_price, 2),
                "exit_price": round(fill, 2),
                "shares": position,
                "pnl": round(pnl, 2),
                "return_pct": round((fill / entry_price - 1) * 100, 2)
                if entry_price
                else 0.0,
                "exit_reason": reason,
            }
        )
        return proceeds

    for i in range(len(df)):
        date = df.index[i]

        # ── 1. Manage an open position: stops/targets are checked intrabar ──
        if position > 0 and use_stops:
            hit_stop = low[i] <= stop_price
            hit_target = high[i] >= target_price

            # When one bar spans both levels, assume the stop filled first.
            # Daily bars carry no intrabar path, and the optimistic assumption
            # is what makes backtests look better than live trading.
            if hit_stop:
                cash += _record_exit(stop_price, date, "Stop loss")
                position = 0
            elif hit_target:
                cash += _record_exit(target_price, date, "Target")
                position = 0

        # ── 2. Act on the PREVIOUS bar's signal at THIS bar's open ──────────
        # signal[i-1] was derived from close[i-1]; the first price actually
        # tradeable after that is open[i].
        if i > 0:
            prev_signal = sig_arr[i - 1]
            fill_open = open_px[i] if np.isfinite(open_px[i]) and open_px[i] > 0 else close[i]

            if prev_signal == 1 and position == 0:
                buy_fill = fill_open * (1 + slippage)
                unit_cost = buy_fill * (1 + commission)
                shares = int(cash / unit_cost) if unit_cost > 0 else 0
                if shares > 0:
                    cash -= shares * unit_cost
                    position = shares
                    entry_price = buy_fill
                    entry_date = date

                    atr = atr_arr[i - 1]
                    if not np.isfinite(atr) or atr <= 0:
                        atr = entry_price * 0.02
                    stop_price = max(0.01, entry_price - stop_mult * atr)
                    target_price = entry_price + target_mult * atr

            elif prev_signal == -1 and position > 0:
                cash += _record_exit(fill_open, date, "Sell signal")
                position = 0

        if position > 0:
            bars_in_market += 1

        equity_values.append(cash + position * close[i])

    # Close anything still open at the final close.
    if position > 0:
        cash += _record_exit(close[-1], df.index[-1], "End of data")
        position = 0
        equity_values[-1] = cash

    equity = float(cash)
    eq_series = pd.Series(equity_values, index=df.index, dtype=float)

    # ── Strategy metrics ────────────────────────────────────────────────────
    net_profit = equity - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100

    n_days = max((df.index[-1] - df.index[0]).days, 1)
    years = n_days / 365.25
    cagr = _cagr(initial_capital, equity, years)

    sharpe, sortino = _annualised_stats(eq_series)
    max_dd = _max_drawdown_pct(eq_series)

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        wins = trade_df[trade_df["pnl"] > 0]
        losses = trade_df[trade_df["pnl"] <= 0]

        win_rate = len(wins) / len(trade_df) * 100
        avg_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0

        gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            # No losing trades: report a capped sentinel rather than 1e13.
            profit_factor = 99.0 if gross_profit > 0 else 0.0
        profit_factor = min(profit_factor, 99.0)

        avg_trade = float(trade_df["pnl"].mean())
        win_frac = win_rate / 100.0
        expectancy = (win_frac * avg_win) - ((1 - win_frac) * abs(avg_loss))
        exit_breakdown = trade_df["exit_reason"].value_counts().to_dict()
    else:
        win_rate = profit_factor = avg_trade = expectancy = 0.0
        exit_breakdown = {}

    # ── Benchmark: buy and hold the same window ─────────────────────────────
    bh_entry = close[0] * (1 + slippage) * (1 + commission)
    bh_shares = int(initial_capital / bh_entry) if bh_entry > 0 else 0
    bh_cash = initial_capital - bh_shares * bh_entry
    bh_curve = pd.Series(bh_cash + bh_shares * close, index=df.index, dtype=float)
    bh_final = bh_cash + bh_shares * close[-1] * (1 - slippage) * (1 - commission)

    benchmark_return_pct = ((bh_final - initial_capital) / initial_capital) * 100
    benchmark_cagr = _cagr(initial_capital, bh_final, years)
    benchmark_max_dd = _max_drawdown_pct(bh_curve)

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
        benchmark_return_pct=round(benchmark_return_pct, 2),
        benchmark_cagr=round(benchmark_cagr, 2),
        benchmark_max_drawdown=round(benchmark_max_dd, 2),
        excess_return_pct=round(net_profit_pct - benchmark_return_pct, 2),
        time_in_market_pct=round(bars_in_market / len(df) * 100, 1),
        exit_breakdown=exit_breakdown,
    )
