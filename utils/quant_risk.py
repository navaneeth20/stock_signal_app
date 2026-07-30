"""
utils/quant_risk.py
===================
Quantitative Risk & Monte Carlo Simulation Engine.
Runs 1,000 price path simulations to compute Probability of Profit (PoP), Expected Value (EV),
and Value at Risk (VaR).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    symbol: str
    num_simulations: int
    horizon_days: int
    win_probability: float      # PoP (0.0 to 100.0 %)
    expected_value: float       # EV in ₹ per trade
    var_95: float               # Value at Risk at 95% confidence (₹)
    median_price: float         # 50th percentile projected price
    p10_price: float            # 10th percentile (bearish case)
    p90_price: float            # 90th percentile (bullish case)
    simulated_paths: np.ndarray  # (num_simulations, horizon_days + 1) matrix for plotting


def run_monte_carlo_simulation(
    df: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    num_simulations: int = 1000,
    horizon_days: int = 20,
) -> MonteCarloResult:
    """
    Run a Monte Carlo simulation based on historical daily log returns.

    Args:
        df: Enriched DataFrame with Close prices.
        entry_price: Entry price for the trade.
        stop_loss: Target stop loss price.
        take_profit: Target take profit price.
        num_simulations: Number of Monte Carlo iterations (default 1000).
        horizon_days: Forward projection horizon in trading days (default 20).

    Returns:
        MonteCarloResult object.
    """
    symbol = df.attrs.get("symbol", "STOCK")
    if len(df) < 30:
        # Fallback if insufficient historical data
        return MonteCarloResult(
            symbol=symbol,
            num_simulations=num_simulations,
            horizon_days=horizon_days,
            win_probability=50.0,
            expected_value=0.0,
            var_95=entry_price * 0.05,
            median_price=entry_price,
            p10_price=entry_price * 0.95,
            p90_price=entry_price * 1.05,
            simulated_paths=np.full((num_simulations, horizon_days + 1), entry_price),
        )

    # Calculate log returns
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    mean_return = returns.mean()
    volatility = returns.std()

    # Drift adjustment for Geometric Brownian Motion
    drift = mean_return - (0.5 * volatility**2)

    # Generate random shocks: shape (num_simulations, horizon_days)
    np.random.seed(42)  # Deterministic seed for reproducible simulations
    random_shocks = np.random.normal(0, 1, (num_simulations, horizon_days))

    # Calculate simulated daily returns
    daily_returns = np.exp(drift + volatility * random_shocks)

    # Build price paths matrix: shape (num_simulations, horizon_days + 1)
    paths = np.zeros((num_simulations, horizon_days + 1))
    paths[:, 0] = entry_price

    for t in range(1, horizon_days + 1):
        paths[:, t] = paths[:, t - 1] * daily_returns[:, t - 1]

    # Evaluate win/loss conditions over paths
    hits_target = np.any(paths >= take_profit, axis=1)
    hits_stop = np.any(paths <= stop_loss, axis=1)

    # Count wins: hit target before stop, or ended above entry
    win_count = 0
    for i in range(num_simulations):
        t_target = np.where(paths[i] >= take_profit)[0]
        t_stop = np.where(paths[i] <= stop_loss)[0]

        first_target = t_target[0] if len(t_target) > 0 else 999
        first_stop = t_stop[0] if len(t_stop) > 0 else 999

        if first_target < first_stop:
            win_count += 1
        elif first_target == 999 and first_stop == 999:
            if paths[i, -1] > entry_price:
                win_count += 1

    win_prob = (win_count / num_simulations) * 100.0

    # Calculate Expected Value (EV) per trade
    profit_amt = take_profit - entry_price
    loss_amt = entry_price - stop_loss
    pop_ratio = win_prob / 100.0
    expected_value = (pop_ratio * profit_amt) - ((1.0 - pop_ratio) * loss_amt)

    # Final price distribution at horizon
    final_prices = paths[:, -1]
    p10 = float(np.percentile(final_prices, 10))
    median = float(np.percentile(final_prices, 50))
    p90 = float(np.percentile(final_prices, 90))

    # Value at Risk (VaR 95%)
    var_95 = float(entry_price - np.percentile(final_prices, 5))

    return MonteCarloResult(
        symbol=symbol,
        num_simulations=num_simulations,
        horizon_days=horizon_days,
        win_probability=round(win_prob, 1),
        expected_value=round(expected_value, 2),
        var_95=round(max(0, var_95), 2),
        median_price=round(median, 2),
        p10_price=round(p10, 2),
        p90_price=round(p90, 2),
        simulated_paths=paths,
    )
