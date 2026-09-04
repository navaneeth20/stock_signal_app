"""
tests/test_signals.py
======================
Unit tests for the signal engine, scoring, risk, backtest and Monte Carlo.

Several tests here are regressions for bugs that were invisible in the UI —
they produced a plausible-looking number rather than an error. Those are
marked with the behaviour they lock down.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import run_backtest
from indicators.bollinger import bollinger_signal, compute_bollinger
from indicators.rsi import compute_rsi, rsi_signal
from indicators.supertrend import compute_supertrend
from strategies.risk import RiskMetrics, calculate_risk
from strategies.scoring import compute_score, label_from_score
from strategies.signal_engine import (
    apply_confidence_modifier,
    compute_all_indicators,
    generate_signal,
    volume_signal,
)
from utils.quant_risk import run_monte_carlo_simulation


def _make_ohlcv(periods: int = 300, seed: int = 42, drift: float = 0.0) -> pd.DataFrame:
    """Synthetic but well-formed OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(seed)
    close = 500.0 + np.cumsum(rng.normal(drift, 5.0, periods))
    close = np.clip(close, 50.0, None)
    high = close + np.abs(rng.normal(0, 3.0, periods))
    low = np.clip(close - np.abs(rng.normal(0, 3.0, periods)), 1.0, None)
    open_p = low + (high - low) * rng.random(periods)
    volume = rng.integers(10_000, 500_000, size=periods)

    return pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def enriched_ohlcv() -> pd.DataFrame:
    return compute_all_indicators(_make_ohlcv())


# ── Scoring ───────────────────────────────────────────────────────────────────

def test_label_from_score():
    assert label_from_score(85.0) == "Strong Buy"
    assert label_from_score(65.0) == "Buy"
    assert label_from_score(50.0) == "Hold"
    assert label_from_score(35.0) == "Sell"
    assert label_from_score(15.0) == "Strong Sell"


def test_compute_score_bounds():
    result = compute_score(
        {
            "ema": {"score": 2}, "rsi": {"score": 2}, "macd": {"score": 2},
            "supertrend": {"score": 2}, "adx": {"score": 1},
            "volume": {"score": 2}, "vwap": {"score": 1},
        }
    )
    assert 0 <= result["confidence"] <= 100


def test_all_neutral_scores_to_fifty():
    neutral = {k: {"score": 0} for k in
               ("ema", "rsi", "macd", "supertrend", "adx", "volume", "vwap")}
    assert compute_score(neutral)["confidence"] == pytest.approx(50.0)


def test_rsi_can_reach_full_bullish_weight():
    """
    Regression: RSI used to cap at +1 while reaching -2, so it could never
    deliver more than 75% of its configured weight — a permanent bearish tilt
    on every stock.
    """
    df = pd.DataFrame({"RSI": [72.0]})
    bullish = rsi_signal(df)
    assert bullish["score"] == 2

    df_bear = pd.DataFrame({"RSI": [28.0]})
    assert rsi_signal(df_bear)["score"] == -2


def test_rsi_mean_reversion_only_when_ranging():
    """High RSI is bullish in a trend, bearish only when ADX says it's ranging."""
    df = pd.DataFrame({"RSI": [75.0]})
    assert rsi_signal(df, adx_value=35.0)["score"] > 0   # trending → momentum
    assert rsi_signal(df, adx_value=15.0)["score"] < 0   # ranging → fade


# ── Volume direction ──────────────────────────────────────────────────────────

def test_volume_signal_is_signed_by_bar_direction():
    """
    Regression: heavy volume scored +1 "strong buying interest" regardless of
    whether the bar was up or down, so an 8% crash on 4x volume read bullish.
    """
    down_bar = pd.DataFrame(
        {"Open": [100.0], "Close": [92.0], "High": [100.0], "Low": [91.0],
         "Volume": [4_000_000], "Volume_Ratio": [4.0]}
    )
    assert volume_signal(down_bar)["score"] < 0

    up_bar = pd.DataFrame(
        {"Open": [92.0], "Close": [100.0], "High": [101.0], "Low": [91.0],
         "Volume": [4_000_000], "Volume_Ratio": [4.0]}
    )
    assert volume_signal(up_bar)["score"] > 0


# ── Bollinger monotonicity ────────────────────────────────────────────────────

def test_bollinger_score_is_monotonic():
    """
    Regression: price below the lower band scored -2 while price just above it
    scored +1, so the sign flipped as price recovered upward through the band.
    """
    def _score_at(bb_pct: float) -> int:
        df = pd.DataFrame(
            {"Close": [100.0], "BB_Upper": [110.0], "BB_Lower": [90.0],
             "BB_Mid": [100.0], "BB_Pct": [bb_pct]}
        )
        return bollinger_signal(df)["score"]

    scores = [_score_at(p) for p in (0.02, 0.15, 0.50, 0.85, 0.98)]
    assert scores == sorted(scores, reverse=True)


# ── Supertrend ────────────────────────────────────────────────────────────────

def test_supertrend_direction_is_binary_and_bands_carry():
    df = compute_supertrend(_make_ohlcv(120), period=10, multiplier=3.0)
    assert set(df["Supertrend_Direction"].unique()) <= {1, -1}
    assert df["Supertrend"].iloc[20:].notna().all()

    # In an uptrend the line sits below price; in a downtrend, above.
    up = df[df["Supertrend_Direction"] == 1]
    if not up.empty:
        assert (up["Supertrend"] <= up["High"]).all()


# ── Signal generation ─────────────────────────────────────────────────────────

def test_generate_signal(enriched_ohlcv):
    res = generate_signal("RELIANCE.NS", enriched_ohlcv)
    assert res.symbol == "RELIANCE.NS"
    assert res.signal in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
    assert 0 <= res.confidence <= 100
    assert res.entry_price > 0
    assert res.risk_reward > 0


def test_generate_signal_levels_match_direction(enriched_ohlcv):
    """
    Regression: the engine hardcoded a long-side stop and target, so a Strong
    Sell showed a stop below entry and a target above it.
    """
    res = generate_signal("TEST.NS", enriched_ohlcv)
    if res.signal in ("Sell", "Strong Sell"):
        assert res.stop_loss > res.entry_price
        assert res.take_profit < res.entry_price
    else:
        assert res.stop_loss < res.entry_price
        assert res.take_profit > res.entry_price


def test_confidence_modifier_relabels():
    """
    Regression: post-hoc confidence adjustments left the label untouched, so
    the header could read "Buy - 82%" when Strong Buy starts at 75.
    """
    from strategies.signal_engine import SignalResult

    result = SignalResult(
        symbol="X", signal=label_from_score(62.0), confidence=62.0,
        entry_price=100.0, stop_loss=97.0, take_profit=106.0, risk_reward=2.0,
    )
    assert result.signal == "Buy"

    apply_confidence_modifier(result, 15.0, "Multi-timeframe alignment")
    assert result.confidence == pytest.approx(77.0)
    assert result.signal == "Strong Buy"        # label followed the number

    apply_confidence_modifier(result, -40.0, "Test downgrade")
    assert result.signal == label_from_score(result.confidence)


def test_confidence_modifier_clamps():
    from strategies.signal_engine import SignalResult

    result = SignalResult(
        symbol="X", signal="Buy", confidence=95.0, entry_price=100.0,
        stop_loss=97.0, take_profit=106.0, risk_reward=2.0,
    )
    apply_confidence_modifier(result, 50.0, "Overflow")
    assert result.confidence == 100.0


# ── Risk ──────────────────────────────────────────────────────────────────────

def test_calculate_risk_long(enriched_ohlcv):
    res = calculate_risk(enriched_ohlcv, "Buy", capital=100_000.0, risk_per_trade=0.02)
    assert isinstance(res, RiskMetrics)
    assert res.direction == 1
    assert res.stop_loss < res.entry_price < res.take_profit
    assert res.max_position_size > 0
    assert res.risk_amount == pytest.approx(2000.0)


def test_calculate_risk_short(enriched_ohlcv):
    res = calculate_risk(enriched_ohlcv, "Strong Sell", capital=100_000.0)
    assert res.direction == -1
    assert res.take_profit < res.entry_price < res.stop_loss


# ── Backtest ──────────────────────────────────────────────────────────────────

def test_run_backtest(enriched_ohlcv):
    bt = run_backtest(enriched_ohlcv, initial_capital=100_000.0)
    assert bt.initial_capital == 100_000.0
    assert isinstance(bt.net_profit, float)
    assert isinstance(bt.win_rate, float)
    assert bt.max_drawdown <= 0
    assert len(bt.equity_curve) == len(enriched_ohlcv)


def test_backtest_reports_benchmark(enriched_ohlcv):
    """A return figure is meaningless without buy-and-hold to compare against."""
    bt = run_backtest(enriched_ohlcv, initial_capital=100_000.0)
    assert bt.excess_return_pct == pytest.approx(
        bt.net_profit_pct - bt.benchmark_return_pct, abs=0.01
    )
    assert 0 <= bt.time_in_market_pct <= 100


def test_backtest_metrics_are_finite(enriched_ohlcv):
    """
    Regression: expectancy was NaN when a run had zero winning trades, and
    profit factor blew up to ~1e13 when it had zero losers.
    """
    bt = run_backtest(enriched_ohlcv, initial_capital=100_000.0)
    for value in (bt.expectancy, bt.profit_factor, bt.sharpe_ratio, bt.sortino_ratio):
        assert np.isfinite(value), f"{value} is not finite"
    assert bt.profit_factor <= 99.0


def test_backtest_respects_stop_loss(enriched_ohlcv):
    bt = run_backtest(enriched_ohlcv, initial_capital=100_000.0, use_stops=True)
    if not bt.trade_log.empty:
        assert "exit_reason" in bt.trade_log.columns
        assert bt.trade_log["exit_reason"].isin(
            {"Stop loss", "Target", "Sell signal", "End of data"}
        ).all()


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def test_monte_carlo_long(enriched_ohlcv):
    mc = run_monte_carlo_simulation(
        df=enriched_ohlcv, entry_price=500.0, stop_loss=480.0, take_profit=540.0,
        num_simulations=500, horizon_days=10, direction=1, seed=7,
    )
    assert mc.num_simulations == 500
    assert 0.0 <= mc.win_probability <= 100.0
    assert mc.var_95 >= 0.0
    assert mc.direction == 1


def test_monte_carlo_short_is_not_zero(enriched_ohlcv):
    """
    Regression: for a short the target sits below entry and the stop above, but
    both comparisons ran in the long direction, so every path registered both
    levels as hit at t=0. Win probability was pinned at exactly 0.0% for every
    Sell and Strong Sell signal — while EV came out positive.
    """
    mc = run_monte_carlo_simulation(
        df=enriched_ohlcv, entry_price=500.0, stop_loss=520.0, take_profit=460.0,
        num_simulations=500, horizon_days=10, direction=-1, seed=7,
    )
    assert mc.direction == -1
    assert 1.0 < mc.win_probability < 99.0, (
        f"Short PoP was {mc.win_probability}%, which means the direction "
        "handling regressed."
    )


def test_monte_carlo_direction_inferred_from_levels(enriched_ohlcv):
    mc = run_monte_carlo_simulation(
        df=enriched_ohlcv, entry_price=500.0, stop_loss=520.0, take_profit=460.0,
        num_simulations=200, horizon_days=10, seed=7,
    )
    assert mc.direction == -1


def test_monte_carlo_ev_is_consistent_with_pop(enriched_ohlcv):
    """
    EV used to assume every win captured the full take-profit, even though
    paths that merely finished above entry counted as wins. It now comes from
    actual per-path exit prices, so it must sit inside the loss/profit range.
    """
    entry, stop, target = 500.0, 480.0, 540.0
    mc = run_monte_carlo_simulation(
        df=enriched_ohlcv, entry_price=entry, stop_loss=stop, take_profit=target,
        num_simulations=500, horizon_days=10, direction=1, seed=11,
    )
    assert (entry - stop) * -1 <= mc.expected_value <= (target - entry)


def test_monte_carlo_seed_is_reproducible(enriched_ohlcv):
    kwargs = dict(
        df=enriched_ohlcv, entry_price=500.0, stop_loss=480.0, take_profit=540.0,
        num_simulations=200, horizon_days=10, direction=1,
    )
    a = run_monte_carlo_simulation(seed=99, **kwargs)
    b = run_monte_carlo_simulation(seed=99, **kwargs)
    assert a.win_probability == b.win_probability


def test_monte_carlo_is_reproducible_without_an_explicit_seed(enriched_ohlcv):
    """
    Same trade, same answer. With no seed the RNG is keyed off the trade
    parameters, so re-running an analysis does not wobble the headline number.
    """
    kwargs = dict(
        df=enriched_ohlcv, entry_price=500.0, stop_loss=480.0, take_profit=540.0,
        num_simulations=300, horizon_days=10, direction=1,
    )
    a = run_monte_carlo_simulation(**kwargs)
    b = run_monte_carlo_simulation(**kwargs)
    assert a.win_probability == b.win_probability
    assert a.expected_value == b.expected_value


def test_monte_carlo_different_trades_get_different_draws(enriched_ohlcv):
    """
    The derived seed must not collapse every stock onto one shared sample set,
    which is what the old global np.random.seed(42) effectively did.
    """
    base = dict(df=enriched_ohlcv, num_simulations=300, horizon_days=10, direction=1)
    a = run_monte_carlo_simulation(entry_price=500.0, stop_loss=480.0, take_profit=540.0, **base)
    b = run_monte_carlo_simulation(entry_price=640.0, stop_loss=610.0, take_profit=700.0, **base)
    assert not np.allclose(a.simulated_paths[:, -1], b.simulated_paths[:, -1])


def test_monte_carlo_reports_sampling_margin(enriched_ohlcv):
    """PoP is an estimate; the margin has to be present and sanely sized."""
    mc = run_monte_carlo_simulation(
        df=enriched_ohlcv, entry_price=500.0, stop_loss=480.0, take_profit=540.0,
        num_simulations=1000, horizon_days=10, direction=1,
    )
    assert mc.pop_margin_pct > 0
    assert mc.pop_margin_pct < 10  # ~±3.1 at p=0.5, n=1000

    # More paths must tighten the estimate.
    coarse = run_monte_carlo_simulation(
        df=enriched_ohlcv, entry_price=500.0, stop_loss=480.0, take_profit=540.0,
        num_simulations=100, horizon_days=10, direction=1,
    )
    assert coarse.pop_margin_pct > mc.pop_margin_pct
