"""
strategies/scoring.py
=====================
Weighted confidence scoring system.
Each indicator contributes a weighted score → 0–100% confidence.
"""

from __future__ import annotations

from config import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    SIGNAL_WEIGHTS,
    STRONG_BUY_THRESHOLD,
    STRONG_SELL_THRESHOLD,
)


def _normalise_indicator_score(raw_score: int, indicator: str) -> float:
    """
    Map a raw indicator score (typically -2 to +2) to a 0–1 bullish fraction.

    Args:
        raw_score:  Raw signed score from the indicator signal function.
        indicator:  Indicator name (for future per-indicator scaling).

    Returns:
        Float in [0, 1] where 0 = max bearish, 1 = max bullish.
    """
    # Clamp to [-2, 2] then rescale
    clamped = max(-2, min(2, raw_score))
    return (clamped + 2) / 4.0  # maps -2→0, 0→0.5, +2→1


def compute_score(indicator_signals: dict[str, dict]) -> dict:
    """
    Compute weighted confidence score from all indicator signals.

    Args:
        indicator_signals: Dict mapping indicator name → signal dict
                           (must contain 'score' key).

    Returns:
        dict with:
          - confidence (float, 0–100): Weighted bullish percentage.
          - scores (dict): Per-indicator weighted contribution.
          - raw_scores (dict): Raw indicator scores.
    """
    weights = SIGNAL_WEIGHTS
    total_weight = sum(weights.values())

    weighted_sum = 0.0
    per_score: dict[str, float] = {}
    raw: dict[str, int] = {}

    for indicator, sig in indicator_signals.items():
        w = weights.get(indicator, 0)
        raw_score = sig.get("score", 0)
        normalised = _normalise_indicator_score(raw_score, indicator)
        contribution = w * normalised
        weighted_sum += contribution
        per_score[indicator] = round(contribution, 2)
        raw[indicator] = raw_score

    confidence = (weighted_sum / total_weight) * 100
    return {
        "confidence": round(confidence, 1),
        "scores": per_score,
        "raw_scores": raw,
    }


def label_from_score(confidence: float) -> str:
    """
    Convert a numeric confidence score to a signal label.

    Args:
        confidence: Float 0–100.

    Returns:
        One of: 'Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'
    """
    if confidence >= STRONG_BUY_THRESHOLD:
        return "Strong Buy"
    elif confidence >= BUY_THRESHOLD:
        return "Buy"
    elif confidence > SELL_THRESHOLD:
        return "Hold"
    elif confidence > STRONG_SELL_THRESHOLD:
        return "Sell"
    else:
        return "Strong Sell"
