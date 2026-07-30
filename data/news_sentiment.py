"""
data/news_sentiment.py
======================
Market Sentiment & News Intelligence Engine (NLP).
Fetches financial news headlines and performs NLP sentiment analysis for Indian equities.
"""

from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Financial Sentiment Dictionary (Domain-specific for Stock Markets)
BULLISH_KEYWORDS = {
    "growth", "profit", "surge", "gain", "bullish", "record", "jump", "rally",
    "outperform", "buy", "upbeat", "expansion", "dividend", "revenue", "order",
    "contract", "upgrade", "target", "strong", "breakout", "acquisition", "high"
}
BEARISH_KEYWORDS = {
    "fall", "drop", "loss", "plunge", "bearish", "decline", "downgrade", "sell",
    "warning", "slump", "investigation", "penalty", "lawsuit", "debt", "risk",
    "slash", "cut", "inflation", "weak", "disappoint", "crash", "low", "concern"
}


@dataclass
class Article:
    title: str
    link: str
    published: str
    sentiment_score: float  # -1.0 to +1.0
    sentiment_label: str    # "Bullish", "Bearish", "Neutral"


@dataclass
class NewsSentimentResult:
    symbol: str
    company_name: str
    overall_score: float      # -1.0 to +1.0 (-100% to +100%)
    sentiment_label: str      # "Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"
    articles: list[Article] = field(default_factory=list)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0


def _analyze_text_sentiment(text: str) -> tuple[float, str]:
    """Calculate sentiment score (-1 to +1) for a news title."""
    words = text.lower().split()
    bull_hits = sum(1 for w in words if w.strip(",.!?\"'") in BULLISH_KEYWORDS)
    bear_hits = sum(1 for w in words if w.strip(",.!?\"'") in BEARISH_KEYWORDS)

    total = bull_hits + bear_hits
    if total == 0:
        return 0.0, "Neutral"

    score = (bull_hits - bear_hits) / max(total, 1)
    if score >= 0.3:
        label = "Bullish"
    elif score <= -0.3:
        label = "Bearish"
    else:
        label = "Neutral"

    return round(score, 2), label


def fetch_news_sentiment(symbol: str, company_name: Optional[str] = None) -> NewsSentimentResult:
    """
    Fetch financial news headlines for a stock via Google News RSS and compute NLP sentiment.

    Args:
        symbol: Stock symbol (e.g., RELIANCE.NS)
        company_name: Optional company name for better news search.

    Returns:
        NewsSentimentResult containing sentiment score, label, and article list.
    """
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")
    query_str = f"{clean_sym} share price news stock" if not company_name else f"{company_name} stock news"
    encoded_query = urllib.parse.quote(query_str)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    articles: list[Article] = []
    bull_cnt, bear_cnt, neu_cnt = 0, 0, 0

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(rss_url, headers=headers, timeout=6)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:6]:
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")[:16]

                score, label = _analyze_text_sentiment(title)

                if label == "Bullish":
                    bull_cnt += 1
                elif label == "Bearish":
                    bear_cnt += 1
                else:
                    neu_cnt += 1

                articles.append(Article(
                    title=title,
                    link=link,
                    published=pub_date,
                    sentiment_score=score,
                    sentiment_label=label,
                ))
    except Exception as exc:
        logger.warning("Failed to fetch news for %s: %s", symbol, exc)

    # Compute overall aggregate score
    total_arts = len(articles)
    if total_arts > 0:
        overall_score = sum(a.sentiment_score for a in articles) / total_arts
    else:
        overall_score = 0.0

    if overall_score >= 0.4:
        overall_label = "Very Bullish"
    elif overall_score >= 0.1:
        overall_label = "Bullish"
    elif overall_score <= -0.4:
        overall_label = "Very Bearish"
    elif overall_score <= -0.1:
        overall_label = "Bearish"
    else:
        overall_label = "Neutral"

    return NewsSentimentResult(
        symbol=symbol,
        company_name=company_name or clean_sym,
        overall_score=round(overall_score, 2),
        sentiment_label=overall_label,
        articles=articles,
        bullish_count=bull_cnt,
        bearish_count=bear_cnt,
        neutral_count=neu_cnt,
    )
