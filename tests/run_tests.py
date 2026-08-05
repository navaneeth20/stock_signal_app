"""
tests/run_tests.py
===================
Standard library test runner for StockSense AI without external test runner dependencies.
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from indicators.adx import compute_adx
from indicators.atr import compute_atr
from indicators.bollinger import compute_bollinger
from indicators.ema import compute_ema
from indicators.macd import compute_macd
from indicators.rsi import compute_rsi
from indicators.supertrend import compute_supertrend
from indicators.vwap import compute_vwap
from strategies.signal_engine import compute_all_indicators, generate_signal
from strategies.scoring import compute_score, label_from_score
from strategies.risk import calculate_risk
from backtesting.backtest import run_backtest
from utils.quant_risk import run_monte_carlo_simulation


class TestIndicators(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(100) * 1.5)
        close = np.clip(close, 10.0, None)
        high = close + np.abs(np.random.randn(100) * 1.0)
        low = close - np.abs(np.random.randn(100) * 1.0)
        open_p = low + (high - low) * np.random.rand(100)
        volume = np.random.randint(1000, 100000, size=100)

        self.df = pd.DataFrame(
            {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    def test_compute_ema(self):
        res = compute_ema(self.df, periods=[20, 50])
        self.assertIn("EMA_20", res.columns)
        self.assertIn("EMA_50", res.columns)

    def test_compute_rsi(self):
        res = compute_rsi(self.df, period=14)
        self.assertIn("RSI", res.columns)
        valid = res["RSI"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all())

    def test_compute_macd(self):
        res = compute_macd(self.df, fast=12, slow=26, signal=9)
        self.assertIn("MACD", res.columns)
        self.assertIn("MACD_Signal", res.columns)
        self.assertIn("MACD_Hist", res.columns)

    def test_compute_supertrend(self):
        res = compute_supertrend(self.df, period=10, multiplier=3.0)
        self.assertIn("Supertrend", res.columns)
        self.assertIn("Supertrend_Direction", res.columns)

    def test_compute_adx(self):
        res = compute_adx(self.df, period=14)
        self.assertIn("ADX", res.columns)

    def test_compute_atr(self):
        res = compute_atr(self.df, period=14)
        self.assertIn("ATR", res.columns)
        self.assertTrue((res["ATR"].dropna() > 0).all())

    def test_compute_all_indicators(self):
        res = compute_all_indicators(self.df)
        for col in ["EMA_20", "RSI", "MACD", "Supertrend", "ADX", "ATR", "BB_Upper", "VWAP", "Volume_SMA"]:
            self.assertIn(col, res.columns)


class TestSignalsAndBacktest(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 500.0 + np.cumsum(np.random.randn(100) * 5.0)
        close = np.clip(close, 50.0, None)
        high = close + np.abs(np.random.randn(100) * 3.0)
        low = close - np.abs(np.random.randn(100) * 3.0)
        open_p = low + (high - low) * np.random.rand(100)
        volume = np.random.randint(10000, 500000, size=100)

        raw_df = pd.DataFrame(
            {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
        self.enriched = compute_all_indicators(raw_df)

    def test_generate_signal(self):
        sig = generate_signal("RELIANCE.NS", self.enriched)
        self.assertEqual(sig.symbol, "RELIANCE.NS")
        self.assertIn(sig.signal, ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"])
        self.assertTrue(0 <= sig.confidence <= 100)

    def test_run_backtest(self):
        bt = run_backtest(self.enriched, initial_capital=100000.0)
        self.assertEqual(bt.initial_capital, 100000.0)
        self.assertIsInstance(bt.net_profit, float)

    def test_monte_carlo(self):
        mc = run_monte_carlo_simulation(
            df=self.enriched,
            entry_price=500.0,
            stop_loss=480.0,
            take_profit=540.0,
            num_simulations=50,
            horizon_days=5,
        )
        self.assertEqual(mc.num_simulations, 50)
        self.assertTrue(0.0 <= mc.win_probability <= 100.0)


class TestUserDatabase(unittest.TestCase):

    def setUp(self):
        from database import initialise_db
        initialise_db()

    def test_user_creation_and_retrieval(self):
        from database import create_or_update_user, get_user_by_email, get_user_by_phone, get_all_users

        user = create_or_update_user("Test Trader", "+91 9999988888", "testtrader@example.com")
        self.assertEqual(user["name"], "Test Trader")
        self.assertEqual(user["email"], "testtrader@example.com")

        by_email = get_user_by_email("testtrader@example.com")
        self.assertIsNotNone(by_email)
        self.assertEqual(by_email["phone"], "+91 9999988888")

        by_phone = get_user_by_phone("+91 9999988888")
        self.assertIsNotNone(by_phone)
        self.assertEqual(by_phone["name"], "Test Trader")

        all_users = get_all_users()
        self.assertTrue(len(all_users) >= 1)


if __name__ == "__main__":
    unittest.main()
