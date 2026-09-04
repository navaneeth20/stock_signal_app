"""
utils/quant_risk.py
===================
Quantitative Risk & Monte Carlo Simulation Engine.
Runs 1,000 price path simulations to compute Probability of Profit (PoP), Expected Value (EV),
and Value at Risk (VaR).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _derive_seed(*parts: object) -> int:
    """
    Build a deterministic 64-bit seed from the trade parameters.

    Uses blake2b rather than hash() because Python randomises string hashing
    per process, which would defeat the point.
    """
    payload = "|".join(f"{p!r}" for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


@dataclass
class MonteCarloResult:
    symbol: str
    num_simulations: int
    horizon_days: int
    win_probability: float      # PoP (0.0 to 100.0 %)
    expected_value: float       # EV in ₹ per share, from actual path exits
    var_95: float               # Value at Risk at 95% confidence (₹)
    median_price: float         # 50th percentile projected price
    p10_price: float            # 10th percentile (bearish case)
    p90_price: float            # 90th percentile (bullish case)
    simulated_paths: np.ndarray  # (num_simulations, horizon_days + 1) matrix for plotting
    direction: int = 1          # +1 = long trade, -1 = short trade
    drift_note: str = ""        # How drift was set, for display

    # 95% sampling margin on win_probability, in percentage points. PoP is an
    # estimate from a finite number of paths, not an exact figure — at 1,000
    # paths and p≈0.45 the margin is roughly ±3. Showing it stops the number
    # reading as more precise than it is.
    pop_margin_pct: float = 0.0


def run_monte_carlo_simulation(
    df: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    num_simulations: int = 1000,
    horizon_days: int = 20,
    direction: Optional[int] = None,
    seed: Optional[int] = None,
    use_historical_drift: bool = False,
    df_of: int = 4,
) -> MonteCarloResult:
    """
    Run a Monte Carlo simulation based on historical daily volatility.

    Drift defaults to zero. Feeding the trailing historical mean return back
    into the projection just restates recent momentum as "probability" — a
    stock up 60% over the sample carries ~0.19%/day of drift, which compounds
    to roughly +4% baked into a 20-day forecast before any randomness. The
    honest question this answers is: given this volatility, how often does the
    target arrive before the stop?

    Shocks are drawn from a Student-t distribution rather than a normal, since
    Indian mid- and smallcap daily returns are visibly fat-tailed and a normal
    understates VaR.

    Known limitation: paths are evaluated at daily closes, so a stop touched
    intraday and recovered by the close is not counted. Win probability is
    therefore mildly optimistic for tight stops.

    Args:
        df:                   Enriched DataFrame with Close prices.
        entry_price:          Entry price for the trade.
        stop_loss:            Stop loss price (below entry for longs, above for shorts).
        take_profit:          Take profit price (above entry for longs, below for shorts).
        num_simulations:      Number of Monte Carlo iterations.
        horizon_days:         Forward projection horizon in trading days.
        direction:            +1 long, -1 short. Inferred from the levels if omitted.
        seed:                 Explicit RNG seed. When omitted, a seed is derived
                              from the trade parameters so that re-running the
                              same analysis reproduces the same figure, while
                              different setups still get independent draws.
                              (The old code called np.random.seed(42) globally,
                              which reset every other RNG in the process and
                              made every stock share one sample path set.)
        use_historical_drift: Opt back in to trailing-mean drift (not recommended).
        df_of:                Degrees of freedom for the t-distribution.

    Returns:
        MonteCarloResult object.
    """
    symbol = df.attrs.get("symbol", "STOCK")

    # Infer trade direction from the level layout when not told.
    if direction is None:
        direction = 1 if take_profit >= entry_price else -1
    direction = 1 if direction >= 0 else -1

    if len(df) < 30 or entry_price <= 0:
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
            direction=direction,
            drift_note="Insufficient history — flat projection.",
        )

    # Daily log returns → volatility
    returns = np.log(df["Close"] / df["Close"].shift(1)).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    volatility = float(returns.std())
    if not np.isfinite(volatility) or volatility <= 0:
        volatility = 0.015  # ~1.5% daily fallback

    if use_historical_drift:
        mu = float(returns.mean())
        drift_note = "Drift: trailing historical mean (extrapolates past momentum)."
    else:
        mu = 0.0
        drift_note = "Drift: zero (risk-neutral). Measures volatility, not momentum."

    drift = mu - (0.5 * volatility**2)

    # Derive a stable seed from the inputs when none is given, so the same
    # trade reproduces the same PoP on a rerun without pinning every stock to
    # one shared sample path.
    if seed is None:
        seed = _derive_seed(
            symbol, entry_price, stop_loss, take_profit, horizon_days, num_simulations
        )
    rng = np.random.default_rng(seed)

    # Student-t shocks, rescaled to unit variance so `volatility` stays the
    # scale parameter. Var(t_v) = v/(v-2) for v > 2.
    raw = rng.standard_t(df_of, size=(num_simulations, horizon_days))
    random_shocks = raw / np.sqrt(df_of / (df_of - 2.0))

    # Build price paths: cumulative product of exp(drift + vol * shock)
    log_steps = drift + volatility * random_shocks
    paths = np.empty((num_simulations, horizon_days + 1), dtype=float)
    paths[:, 0] = entry_price
    paths[:, 1:] = entry_price * np.exp(np.cumsum(log_steps, axis=1))

    # ── Evaluate each path: which level is touched first? ────────────────────
    # Direction matters. For a long, the target is above and the stop below;
    # for a short both comparisons invert. Comparing in the long direction only
    # made every short register both levels as hit at t=0, which pinned the
    # win probability at exactly 0%.
    future = paths[:, 1:]  # exclude t=0, where price is entry by construction

    if direction == 1:
        target_hits = future >= take_profit
        stop_hits = future <= stop_loss
    else:
        target_hits = future <= take_profit
        stop_hits = future >= stop_loss

    never = horizon_days + 1
    first_target = np.where(target_hits.any(axis=1), target_hits.argmax(axis=1), never)
    first_stop = np.where(stop_hits.any(axis=1), stop_hits.argmax(axis=1), never)

    final_prices = paths[:, -1]

    hit_target_first = first_target < first_stop
    hit_stop_first = first_stop < first_target
    # Both untouched (or touched on the same bar — treat as the adverse case).
    undecided = ~hit_target_first & ~hit_stop_first

    if direction == 1:
        drifted_win = undecided & (final_prices > entry_price)
    else:
        drifted_win = undecided & (final_prices < entry_price)

    wins = hit_target_first | drifted_win
    win_prob = float(wins.mean() * 100.0)

    # 95% sampling margin: 1.96 * sqrt(p(1-p)/n), in percentage points.
    p = win_prob / 100.0
    pop_margin = (
        float(1.96 * np.sqrt(p * (1 - p) / num_simulations) * 100.0)
        if num_simulations > 0
        else 0.0
    )

    # ── Expected value from actual per-path exits ────────────────────────────
    # The old formula assumed every "win" captured the full take-profit, even
    # though paths that merely finished a rupee above entry counted as wins.
    exit_prices = np.where(
        hit_target_first,
        take_profit,
        np.where(hit_stop_first, stop_loss, final_prices),
    )
    pnl_per_share = (exit_prices - entry_price) * direction
    expected_value = float(pnl_per_share.mean())

    p10 = float(np.percentile(final_prices, 10))
    median = float(np.percentile(final_prices, 50))
    p90 = float(np.percentile(final_prices, 90))

    # VaR 95%: the 5th-percentile loss on the position, in the trade's own
    # direction (a short loses when price rises).
    var_95 = float(-np.percentile(pnl_per_share, 5))

    return MonteCarloResult(
        symbol=symbol,
        num_simulations=num_simulations,
        horizon_days=horizon_days,
        win_probability=round(win_prob, 1),
        expected_value=round(expected_value, 2),
        var_95=round(max(0.0, var_95), 2),
        median_price=round(median, 2),
        p10_price=round(p10, 2),
        p90_price=round(p90, 2),
        simulated_paths=paths,
        direction=direction,
        drift_note=drift_note,
        pop_margin_pct=round(pop_margin, 1),
    )
