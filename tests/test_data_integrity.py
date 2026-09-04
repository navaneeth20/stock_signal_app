"""
tests/test_data_integrity.py
=============================
Tests that the app does not present invented data as fact.

The institutional panel used to derive FII/DII "net buying" from price
momentum, split institutional holding with a hardcoded 52/48 ratio, convert
that into a rupee-crore flow figure, and return the same four fund names for
every stock in the market. These tests make sure none of that comes back.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

import data.institutional_flows as flows
from data.institutional_flows import AccumulationProxyResult, fetch_accumulation_proxy
from data.news_sentiment import _analyze_text_sentiment, _strip_publisher


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Never hit yfinance in tests."""
    class _FakeTicker:
        def __init__(self, *_a, **_kw):
            self.info = {}

    monkeypatch.setattr(flows.yf, "Ticker", _FakeTicker)


def _frame(n: int = 60, trend: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(trend, 1.5, n))
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )


# ── No fabricated ownership ───────────────────────────────────────────────────

def test_no_hardcoded_holder_names_anywhere():
    """
    Regression: top_fii_holders was a literal list — Vanguard, BlackRock,
    Norway's GPFG, Fidelity — returned for every stock the user looked up.
    """
    source = (
        open(flows.__file__, encoding="utf-8").read()
        # The module docstring explains what was removed; strip it before
        # searching so the explanation doesn't trip the test.
        .split('"""', 2)[-1]
    )
    for name in ("Vanguard", "BlackRock", "Fidelity", "Nippon", "SBI Mutual"):
        assert name not in source, f"Hardcoded holder name '{name}' is back"


def test_result_exposes_no_fii_dii_split():
    result = fetch_accumulation_proxy("TEST.NS", _frame())
    assert isinstance(result, AccumulationProxyResult)
    for banned in ("fii_holding_pct", "mf_dii_holding_pct", "fii_est_flow_cr",
                   "mf_est_flow_cr", "estimated_30d_flow_cr", "top_fii_holders",
                   "top_mf_holders", "promoter_holding_pct"):
        assert not hasattr(result, banned), f"{banned} should no longer exist"


def test_missing_ownership_data_is_reported_as_missing():
    """
    Regression: absent yfinance fields fell back to 25% institutional / 50%
    insider, so the app confidently reported a shareholding structure for a
    company it knew nothing about.
    """
    result = fetch_accumulation_proxy("TEST.NS", _frame())
    assert result.ownership_available is False
    assert result.total_institutional_pct is None
    assert result.insider_pct is None


def test_disclaimer_is_attached():
    result = fetch_accumulation_proxy("TEST.NS", _frame())
    assert "not shareholding data" in result.disclaimer.lower()


def test_proxy_direction_tracks_buying_pressure():
    up = fetch_accumulation_proxy("UP.NS", _frame(trend=1.2))
    down = fetch_accumulation_proxy("DOWN.NS", _frame(trend=-1.2))
    assert up.accumulation_score > down.accumulation_score
    assert -100 <= up.accumulation_score <= 100
    assert -100 <= down.accumulation_score <= 100


def test_proxy_handles_empty_frame():
    result = fetch_accumulation_proxy("TEST.NS", pd.DataFrame())
    assert result.accumulation_score == 0.0
    assert result.evidence


# ── News sentiment ────────────────────────────────────────────────────────────

def test_publisher_suffix_is_stripped():
    assert _strip_publisher("Reliance hits record high - Economic Times") == (
        "Reliance hits record high"
    )


@pytest.mark.parametrize("headline,expected", [
    ("Profit falls 30% in Q3", "Bearish"),
    ("No growth seen in FY26", "Bearish"),
    ("Revenue rises as margins expand", "Bullish"),
    ("Shares surge on strong order book", "Bullish"),
    ("Stock tumbles after downgrade", "Bearish"),
])
def test_negation_is_handled(headline, expected):
    """
    Regression: bag-of-words with no negation read "Profit falls 30%" as
    neutral (profit +1, falls -1), and "no growth" as bullish.
    """
    _, label = _analyze_text_sentiment(headline)
    assert label == expected, f"{headline!r} scored {label}, expected {expected}"


def test_reduction_headlines_are_not_scored_bearish():
    """
    "Company cuts debt by half" used to score -1.0 Bearish, because both
    "cuts" and "debt" sat in the bearish dictionary. Removing the subject
    nouns fixes the false negative.

    It lands on Neutral rather than Bullish: recognising that reducing a bad
    thing is good news needs phrase-level semantics that a keyword bag cannot
    express. The module docstring points at FinBERT for that. Neutral is the
    honest answer here, and it is the improvement being locked in.
    """
    score, label = _analyze_text_sentiment("Company cuts debt by half")
    assert label != "Bearish"
    assert score >= 0.0


def test_neutral_subject_nouns_do_not_score():
    """'Revenue', 'debt' and 'target' describe a topic, not a sentiment."""
    score, label = _analyze_text_sentiment("Company announces revenue target and debt plan")
    assert label == "Neutral"
    assert score == 0.0
