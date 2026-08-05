"""
tests/test_indicators.py
========================
Unit test suite for technical indicator computations.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from indicators.adx import compute_adx
from indicators.atr import compute_atr
from indicators.bollinger import compute_bollinger
from indicators.ema import compute_ema
from indicators.macd import compute_macd
from indicators.rsi import compute_rsi
from indicators.supertrend import compute_supertrend
from indicators.vwap import compute_vwap
from strategies.signal_engine import compute_all_indicators


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 100 bars of synthetic OHLCV data."""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(100) * 1.5)
    close = np.clip(close, 10.0, None)
    high = close + np.abs(np.random.randn(100) * 1.0)
    low = close - np.abs(np.random.randn(100) * 1.0)
    open_p = low + (high - low) * np.random.rand(100)
    volume = np.random.randint(1000, 100000, size=100)

    df = pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


def test_compute_ema(sample_ohlcv):
    df = compute_ema(sample_ohlcv, periods=[20, 50])
    assert "EMA_20" in df.columns
    assert "EMA_50" in df.columns
    assert not df["EMA_20"].dropna().empty


def test_compute_rsi(sample_ohlcv):
    df = compute_rsi(sample_ohlcv, period=14)
    assert "RSI" in df.columns
    valid_rsi = df["RSI"].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_compute_macd(sample_ohlcv):
    df = compute_macd(sample_ohlcv, fast=12, slow=26, signal=9)
    assert "MACD" in df.columns
    assert "MACD_Signal" in df.columns
    assert "MACD_Hist" in df.columns


def test_compute_supertrend(sample_ohlcv):
    df = compute_supertrend(sample_ohlcv, period=10, multiplier=3.0)
    assert "Supertrend" in df.columns
    assert "Supertrend_Direction" in df.columns
    dirs = df["Supertrend_Direction"].dropna().unique()
    assert set(dirs).issubset({1, -1, 0})


def test_compute_adx(sample_ohlcv):
    df = compute_adx(sample_ohlcv, period=14)
    assert "ADX" in df.columns
    assert "+DI" in df.columns
    assert "-DI" in df.columns


def test_compute_atr(sample_ohlcv):
    df = compute_atr(sample_ohlcv, period=14)
    assert "ATR" in df.columns
    assert (df["ATR"].dropna() > 0).all()


def test_compute_bollinger(sample_ohlcv):
    df = compute_bollinger(sample_ohlcv, period=20, std_dev=2.0)
    assert "BB_Upper" in df.columns
    assert "BB_Middle" in df.columns
    assert "BB_Lower" in df.columns
    valid = df.dropna(subset=["BB_Upper", "BB_Middle", "BB_Lower"])
    assert (valid["BB_Upper"] >= valid["BB_Middle"]).all()
    assert (valid["BB_Middle"] >= valid["BB_Lower"]).all()


def test_compute_all_indicators(sample_ohlcv):
    df = compute_all_indicators(sample_ohlcv)
    required_cols = ["EMA_20", "RSI", "MACD", "Supertrend", "ADX", "ATR", "BB_Upper", "VWAP", "Volume_SMA"]
    for col in required_cols:
        assert col in df.columns
