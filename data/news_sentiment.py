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
#
# Only words that carry sentiment on their own belong here. Neutral subject
# nouns — "revenue", "debt", "target", "order" — were removed: they describe
# what a headline is about, not whether it is good news. Leaving them in made
# "Company cuts debt by half" read as doubly bearish.
BULLISH_KEYWORDS = {
    "growth", "surge", "surges", "gain", "gains", "bullish",
    "record", "jump", "jumps", "rally", "rallies", "outperform", "upbeat",
    "expansion", "upgrade", "upgrades", "strong", "strength", "breakout",
    "beats", "beat", "soars", "soar", "rises", "rise", "wins", "win", "boost",
    "boosts", "higher", "outperforms", "robust", "recovery", "rebound",
}

# "profit" belongs with "revenue" and "debt": it names what the headline is
# about, not whether the news is good. The verb carries the sentiment —
# "profit rises" is bullish, "profit falls" is bearish. Scoring the noun as
# bullish made "Profit falls 30%" cancel out to Neutral.
BEARISH_KEYWORDS = {
    "fall", "falls", "drop", "drops", "loss", "losses", "plunge", "plunges",
    "bearish", "decline", "declines", "downgrade", "downgrades", "warning",
    "warns", "slump", "slumps", "investigation", "probe", "penalty", "fine",
    "lawsuit", "fraud", "default", "slash", "slashes", "weak", "weakness",
    "disappoint", "disappoints", "crash", "concern", "concerns", "misses",
    "miss", "lower", "sinks", "sink", "tumbles", "tumble", "underperform",
    "downturn", "layoffs", "resigns", "halt",
}

# Flipping a term's polarity. "Profit falls" and "no growth" are not bullish.
NEGATORS = {
    "no", "not", "never", "without", "fails", "fail", "failed", "lacks",
    "lack", "denies", "deny", "denied", "unlikely", "despite", "wont",
    "cannot", "cant", "halts", "halted", "less", "fewer", "misses",
}
NEGATION_WINDOW = 3  # tokens after a negator that get flipped


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


def _strip_publisher(title: str) -> str:
    """
    Remove the ' - Publisher' suffix Google News RSS appends to every title.

    Without this, publisher names leak into the token stream and can trip
    keyword matches ("Business Standard", "Mint", "The Hindu BusinessLine").
    """
    if " - " in title:
        head, _, tail = title.rpartition(" - ")
        # Only strip when the tail looks like a masthead, not part of the story.
        if head and len(tail.split()) <= 5:
            return head.strip()
    return title.strip()


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens with punctuation stripped."""
    return [w.strip(",.!?\"'();:—–-") for w in text.lower().split()]


def _analyze_text_sentiment(text: str) -> tuple[float, str]:
    """
    Calculate sentiment score (-1 to +1) for a news title.

    Handles negation: a sentiment word within NEGATION_WINDOW tokens after a
    negator has its polarity flipped. So "profit falls" scores bearish rather
    than cancelling to neutral, and "no growth" is not read as bullish.
    """
    words = _tokenise(_strip_publisher(text))

    bull_hits = 0.0
    bear_hits = 0.0
    negate_until = -1

    for idx, word in enumerate(words):
        if not word:
            continue

        if word in NEGATORS:
            negate_until = idx + NEGATION_WINDOW
            continue

        negated = idx <= negate_until

        if word in BULLISH_KEYWORDS:
            if negated:
                bear_hits += 1
            else:
                bull_hits += 1
        elif word in BEARISH_KEYWORDS:
            if negated:
                bull_hits += 1
            else:
                bear_hits += 1

    total = bull_hits + bear_hits
    if total == 0:
        return 0.0, "Neutral"

    score = (bull_hits - bear_hits) / total
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
            seen_titles: set[str] = set()

            # 20 headlines rather than 6 — a handful of titles is too thin a
            # base to call a sentiment reading.
            for item in root.findall(".//item")[:20]:
                raw_title = item.findtext("title", "").strip()
                if not raw_title:
                    continue

                title = _strip_publisher(raw_title)
                dedup_key = title.lower()
                if dedup_key in seen_titles:
                    continue
                seen_titles.add(dedup_key)

                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")[:16]

                score, label = _analyze_text_sentiment(raw_title)

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
