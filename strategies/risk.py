"""
strategies/risk.py
==================
Risk management calculations:
  - Entry, Stop Loss, Take Profit
  - ATR-based stops
  - Position sizing
  - Risk-Reward ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    ATR_STOP_MULTIPLIER,
    DEFAULT_CAPITAL,
    DEFAULT_RISK_PER_TRADE,
    TAKE_PROFIT_RR,
)


@dataclass
class RiskMetrics:
    """Risk management output for a trade."""

    entry_price: float
    stop_loss: float
    take_profit: float
    atr_stop: float
    risk_reward: float
    max_position_size: int    # Number of shares
    capital_allocation: float  # ₹ amount to invest
    risk_amount: float         # ₹ risk per trade
    stop_pct: float            # Stop distance as % of entry


def calculate_risk(
    df: pd.DataFrame,
    signal: str,
    capital: float = DEFAULT_CAPITAL,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    atr_multiplier: float = ATR_STOP_MULTIPLIER,
    rr_ratio: float = TAKE_PROFIT_RR,
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics for a trade.

    Args:
        df:             Enriched DataFrame (must have ATR column).
        signal:         Signal label (affects direction).
        capital:        Total trading capital in ₹.
        risk_per_trade: Fraction of capital to risk (e.g. 0.02 = 2%).
        atr_multiplier: ATR multiplier for stop loss.
        rr_ratio:       Desired Risk:Reward ratio.

    Returns:
        RiskMetrics dataclass.
    """
    last = df.iloc[-1]
    entry = float(last["Close"])
    atr = float(last.get("ATR", entry * 0.02))

    is_long = signal in ("Strong Buy", "Buy", "Hold")
    atr_stop = atr * atr_multiplier

    if is_long:
        stop_loss = entry - atr_stop
        take_profit = entry + atr_stop * rr_ratio
    else:
        stop_loss = entry + atr_stop
        take_profit = entry - atr_stop * rr_ratio

    stop_distance = abs(entry - stop_loss)
    stop_pct = stop_distance / entry

    # Position sizing (fixed fractional)
    risk_amount = capital * risk_per_trade
    max_shares = int(risk_amount / max(stop_distance, 0.01))
    capital_allocation = min(max_shares * entry, capital)
    risk_reward = (abs(take_profit - entry) / stop_distance) if stop_distance > 0 else 0

    return RiskMetrics(
        entry_price=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        atr_stop=round(atr_stop, 2),
        risk_reward=round(risk_reward, 2),
        max_position_size=max_shares,
        capital_allocation=round(capital_allocation, 2),
        risk_amount=round(risk_amount, 2),
        stop_pct=round(stop_pct * 100, 2),
    )
