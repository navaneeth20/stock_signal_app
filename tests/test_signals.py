"""
tests/test_signals.py
======================
Unit test suite for signal engine, confidence scoring, risk calculation, backtest, and Monte Carlo.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest import run_backtest
from strategies.risk import calculate_risk
from strategies.scoring import compute_score, label_from_score
from strategies.signal_engine import compute_all_indicators, generate_signal
from utils.quant_risk import run_monte_carlo_simulation


@pytest.fixture
def enriched_ohlcv() -> pd.DataFrame:
    """Generate 100 bars of enriched OHLCV data."""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 500.0 + np.cumsum(np.random.randn(100) * 5.0)
    close = np.clip(close, 50.0, None)
    high = close + np.abs(np.random.randn(100) * 3.0)
    low = close - np.abs(np.random.randn(100) * 3.0)
    open_p = low + (high - low) * np.random.rand(100)
    volume = np.random.randint(10000, 500000, size=100)

    df = pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return compute_all_indicators(df)


def test_label_from_score():
    assert label_from_score(85.0) == "Strong Buy"
    assert label_from_score(65.0) == "Buy"
    assert label_from_score(50.0) == "Hold"
    assert label_from_score(35.0) == "Sell"
    assert label_from_score(15.0) == "Strong Sell"


def test_compute_score():
    dummy_signals = {
        "ema": {"score": 2},
        "rsi": {"score": 1},
        "macd": {"score": 2},
        "supertrend": {"score": 2},
        "adx": {"score": 1},
        "volume": {"score": 1},
        "vwap": {"score": 1},
    }
    result = compute_score(dummy_signals)
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 100


def test_generate_signal(enriched_ohlcv):
    res = generate_signal("RELIANCE.NS", enriched_ohlcv)
    assert res.symbol == "RELIANCE.NS"
    assert res.signal in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell")
    assert 0 <= res.confidence <= 100
    assert res.entry_price > 0
    assert res.stop_loss < res.entry_price
    assert res.take_profit > res.entry_price
    assert res.risk_reward > 0


def test_calculate_risk():
    res = calculate_risk(entry_price=1000.0, stop_loss=950.0, take_profit=1100.0, capital=100000.0, risk_pct=2.0)
    assert res["position_size"] > 0
    assert res["shares"] == int(100000.0 * 0.02 / 50.0)
    assert res["risk_reward"] == 2.0


def test_run_backtest(enriched_ohlcv):
    bt_res = run_backtest(enriched_ohlcv, initial_capital=100000.0)
    assert bt_res.initial_capital == 100000.0
    assert isinstance(bt_res.net_profit, float)
    assert isinstance(bt_res.win_rate, float)
    assert isinstance(bt_res.max_drawdown, float)


def test_run_monte_carlo_simulation(enriched_ohlcv):
    mc_res = run_monte_carlo_simulation(
        df=enriched_ohlcv,
        entry_price=500.0,
        stop_loss=480.0,
        take_profit=540.0,
        num_simulations=100,
        horizon_days=10,
    )
    assert mc_res.num_simulations == 100
    assert 0.0 <= mc_res.win_probability <= 100.0
    assert mc_res.var_95 >= 0.0
