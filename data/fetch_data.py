"""
data/fetch_data.py
==================
Fetches OHLCV data from Yahoo Finance with retry logic, caching,
and symbol normalisation for Indian exchanges (NSE/BSE).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import yfinance as yf

from data.cache import DataCache

logger = logging.getLogger(__name__)

_cache = DataCache()


class SymbolNotFoundError(ValueError):
    """
    The ticker does not resolve on Yahoo Finance.

    Distinct from a transient network error: retrying will not help, so
    fetch_ohlcv fails fast instead of sleeping through its backoff schedule.
    """


# Daily and weekly bars change once per session — refetching them every five
# minutes is pure waste. Intraday intervals keep the short TTL.
_INTERVAL_TTL_SECONDS: dict[str, int] = {
    "1mo": 86_400,
    "1wk": 43_200,
    "1d": 21_600,   # 6 hours
    "1h": 900,
    "30m": 600,
    "15m": 300,
    "5m": 180,
    "1m": 60,
}


def cache_ttl_for_interval(interval: str) -> int:
    """Return an appropriate cache TTL in seconds for a bar interval."""
    return _INTERVAL_TTL_SECONDS.get(interval, 300)


SYMBOL_ALIASES: dict[str, str] = {
    "HERO": "HEROMOTOCO.NS",
    "HERO.NS": "HEROMOTOCO.NS",
    "HERO.BO": "HEROMOTOCO.BO",
    "HEROMOTO": "HEROMOTOCO.NS",
    "HEROMOTOCORP": "HEROMOTOCO.NS",
    # Tata Motors demerged; the listed passenger-vehicle entity is TMPV.
    "TATAMOTOR": "TMPV.NS",
    "TATAMOTOR.NS": "TMPV.NS",
    "TATAMOTORS": "TMPV.NS",
    "TATAMOTORS.NS": "TMPV.NS",
    # Zomato renamed to Eternal Ltd.
    "ZOMATO": "ETERNAL.NS",
    "ZOMATO.NS": "ETERNAL.NS",
    # Correct NSE symbol for Indian Hotels is INDHOTEL, not INDIHOTEL.
    "INDIHOTEL": "INDHOTEL.NS",
    "INDIHOTEL.NS": "INDHOTEL.NS",
    "INDIANHOTELS": "INDHOTEL.NS",
    "BAJAJAUTO": "BAJAJ-AUTO.NS",
    "BAJAJAUTO.NS": "BAJAJ-AUTO.NS",
    "MM": "M&M.NS",
    "MM.NS": "M&M.NS",
    "MAHINDRA": "M&M.NS",
    "LARSEN": "LT.NS",
    "L&T": "LT.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "KOTAK": "KOTAKBANK.NS",
    "AXIS": "AXISBANK.NS",
    "SBI": "SBIN.NS",
    "ULTRATECH": "ULTRACEMCO.NS",
    "SUN": "SUNPHARMA.NS",
    "REDDY": "DRREDDY.NS",
    "DRREDDYS": "DRREDDY.NS",
    "APOLLO": "APOLLOHOSP.NS",
    "ADANI": "ADANIENT.NS",
    "ADANIPORT": "ADANIPORTS.NS",
    "TATACONSUMER": "TATACONSUM.NS",
    "BAJAJFINANCE": "BAJFINANCE.NS",
    "BAJAJFINSERV": "BAJAJFINSV.NS",
    "PERSISTENTSYS": "PERSISTENT.NS",
    "REC": "RECLTD.NS",
    "REC.NS": "RECLTD.NS",
    "HINDUNILEVER": "HINDUNILVR.NS",
    "HUL": "HINDUNILVR.NS",
    "NESTLE": "NESTLEIND.NS",
    "IONEXCHANGE": "IONEXCHANG.NS",
    "IONEXCHANGE.NS": "IONEXCHANG.NS",
    "IONEXCHANGE.BO": "IONEXCHANG.BO",
    "ION EXCHANGE": "IONEXCHANG.NS",
    "ION EXCHANGE.NS": "IONEXCHANG.NS",
    "ION EXCHANGE.BO": "IONEXCHANG.BO",
    "IONEXCHANG": "IONEXCHANG.NS",
    "IONEXCHANG.NS": "IONEXCHANG.NS",
    "FORCEMOTORS": "FORCEMOTOR.NS",
    "EICHERMOTORS": "EICHERMOT.NS",
}


def normalise_symbol(symbol: str, exchange: str = "NSE") -> str:
    """
    Ensure the symbol has the correct Yahoo Finance suffix and resolve aliases.

    Args:
        symbol: Raw stock symbol e.g. 'HERO', 'ION EXCHANGE', 'RELIANCE.NS'
        exchange: 'NSE' or 'BSE'

    Returns:
        Yahoo Finance formatted symbol e.g. 'IONEXCHANG.NS', 'HEROMOTOCO.NS'
    """
    symbol_upper = symbol.upper().strip()

    if symbol_upper in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[symbol_upper]

    no_space = symbol_upper.replace(" ", "").replace("-", "")
    if no_space in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[no_space]

    clean_sym = symbol_upper.split(".")[0].replace(" ", "")
    if clean_sym in SYMBOL_ALIASES:
        target = SYMBOL_ALIASES[clean_sym]
        suffix = ".BO" if exchange.upper() == "BSE" or symbol_upper.endswith(".BO") else ".NS"
        return target.split(".")[0] + suffix

    suffix_map = {"NSE": ".NS", "BSE": ".BO"}
    suffix = suffix_map.get(exchange.upper(), ".NS")

    if "." not in symbol_upper:
        return f"{clean_sym}{suffix}"
    return f"{clean_sym}{suffix}" if not (symbol_upper.endswith(".NS") or symbol_upper.endswith(".BO")) else symbol_upper.replace(" ", "")



def strip_ns_suffix(symbol: str) -> str:
    """Strip .NS or .BO suffix to get clean ticker symbol (e.g. 'HEROMOTOCO', 'WIPRO')."""
    return symbol.replace(".NS", "").replace(".BO", "").replace(".ns", "").replace(".bo", "").strip()


def fetch_ohlcv(
    symbol: str,
    interval: str = "1d",
    period: str = "1y",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given symbol from Yahoo Finance.

    Args:
        symbol:   Yahoo Finance symbol (e.g. 'RELIANCE.NS').
        interval: Data interval ('1d', '1wk', '1h', '15m', etc.)
        period:   Lookback period string ('1y', '6mo', '3mo') — used when
                  start/end are not provided.
        start:    Start datetime (optional).
        end:      End datetime (optional).
        retries:  Number of retry attempts on transient failure.
        backoff:  Exponential backoff multiplier.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] indexed by
        datetime, sorted ascending.

    Raises:
        ValueError: If the returned data is empty.
    """
    # Ensure symbol is normalised
    symbol = normalise_symbol(symbol)

    cache_key = f"{symbol}_{interval}_{period}_{start}_{end}"
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache HIT for %s", cache_key)
        return cached

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        is_last_attempt = attempt == retries
        try:
            logger.info(
                "Fetching %s | interval=%s | attempt=%d", symbol, interval, attempt
            )
            ticker = yf.Ticker(symbol)

            if start and end:
                df = ticker.history(
                    interval=interval,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=False,
                )
            else:
                df = ticker.history(
                    interval=interval,
                    period=period,
                    auto_adjust=False,
                )

            if df.empty:
                # A ticker that returns nothing is not a transient failure —
                # retrying it three times with backoff just burns ~7 seconds
                # per bad symbol, which adds up fast in a 90-symbol scan.
                raise SymbolNotFoundError(
                    f"No data returned for symbol '{symbol}'. "
                    "Please verify the ticker symbol (e.g. HEROMOTOCO for Hero MotoCorp, TATAMOTORS, RELIANCE)."
                )


            # Standardise column names & data types
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.sort_index(inplace=True)

            # Clean numeric columns and fill missing values
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

            # Drop rows where Close is missing completely if ffill fails
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df.dropna(subset=["Close"], inplace=True)

            if df.empty:
                raise SymbolNotFoundError(f"No valid OHLCV rows for symbol '{symbol}'.")

            _cache.set(cache_key, df, ttl=cache_ttl_for_interval(interval))
            logger.info("Fetched %d rows for %s", len(df), symbol)
            return df

        except SymbolNotFoundError:
            # Permanent: the symbol does not resolve. Fail immediately.
            raise

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if is_last_attempt:
                logger.warning("Final attempt failed for %s: %s", symbol, exc)
                break
            wait = backoff ** attempt
            logger.warning(
                "Attempt %d failed for %s: %s. Retrying in %.1fs…",
                attempt,
                symbol,
                exc,
                wait,
            )
            time.sleep(wait)

    raise ValueError(
        f"Could not fetch '{symbol}' from Yahoo Finance after {retries} attempts. "
        "Please check the ticker symbol (e.g. HEROMOTOCO for Hero MotoCorp)."
    ) from last_exc



def fetch_multiple_stocks(
    symbols: list[str],
    interval: str = "1d",
    period: str = "6mo",
    max_workers: int = 8,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple symbols concurrently (used by the scanner).

    Fetching 90+ symbols one at a time is the single slowest thing the app
    does. Each fetch is almost entirely network wait, so a modest thread pool
    cuts a two-minute scan to well under twenty seconds. Workers are capped at
    8 to stay clear of Yahoo's rate limiting.

    Args:
        symbols:           List of Yahoo Finance symbols.
        interval:          Data interval.
        period:            Lookback period.
        max_workers:       Concurrent fetches. Raise cautiously — Yahoo throttles.
        progress_callback: Called as (completed, total, symbol) after each result.

    Returns:
        Dict mapping symbol → DataFrame (empty DataFrame on failure).
    """
    results: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    if total == 0:
        return results

    def _fetch_one(sym: str) -> tuple[str, pd.DataFrame]:
        try:
            return sym, fetch_ohlcv(sym, interval=interval, period=period)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", sym, exc)
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in symbols}
        for done, future in enumerate(as_completed(futures), start=1):
            sym, df = future.result()
            results[sym] = df
            if progress_callback:
                progress_callback(done, total, sym)

    # Preserve the caller's ordering.
    return {s: results.get(s, pd.DataFrame()) for s in symbols}


def get_company_info(symbol: str) -> dict:
    """
    Return basic company metadata from Yahoo Finance.

    Args:
        symbol: Yahoo Finance symbol.

    Returns:
        Dict with keys: name, sector, industry, marketCap, pe, eps.
    """
    try:
        info = yf.Ticker(symbol).info
        return {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "marketCap": info.get("marketCap", 0),
            "pe": info.get("trailingPE", None),
            "eps": info.get("trailingEps", None),
            "52wHigh": info.get("fiftyTwoWeekHigh", None),
            "52wLow": info.get("fiftyTwoWeekLow", None),
            "beta": info.get("beta", None),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch info for %s: %s", symbol, exc)
        return {}


def get_stock_name(symbol: str) -> str:
    """Get the clean official company name for any stock symbol."""
    from config import NIFTY50_STOCKS, NIFTY_MIDCAP, SECTOR_POOLS

    clean_target = symbol.split(".")[0].upper().strip()

    # 1. Static dict check
    for stock in NIFTY50_STOCKS + NIFTY_MIDCAP:
        clean_stock = stock["symbol"].split(".")[0].upper().strip()
        if clean_stock == clean_target:
            return stock["name"]

    # 2. Sector pools check
    for sector, pool in SECTOR_POOLS.items():
        for stock in pool:
            clean_stock = stock["symbol"].split(".")[0].upper().strip()
            if clean_stock == clean_target:
                return stock["name"]

    # 3. Yahoo Finance dynamic info lookup
    info = get_company_info(symbol)
    if info.get("name") and info["name"] != symbol:
        return info["name"]

    return symbol.replace(".NS", "").replace(".BO", "")


def get_sector_peers(symbol: str) -> tuple[str, list[dict[str, str]]]:
    """Find sector category and list of peer stocks for a given symbol."""
    from config import NIFTY50_STOCKS, SECTOR_POOLS

    clean_target = symbol.split(".")[0].upper().strip()

    # 1. Direct match in sector pools
    for sector_name, pool in SECTOR_POOLS.items():
        if any(s["symbol"].split(".")[0].upper().strip() == clean_target for s in pool):
            peers = [s for s in pool if s["symbol"].split(".")[0].upper().strip() != clean_target]
            return sector_name, peers

    # 2. Dynamic Yahoo Finance sector mapping fallback
    info = get_company_info(symbol)
    yf_sector = info.get("sector", "").lower()

    sector_mapping = {
        "technology": "IT & Software",
        "financial services": "Banking & Financials",
        "financial": "Banking & Financials",
        "automotive": "Automobiles",
        "consumer cyclical": "Automobiles",
        "healthcare": "Pharma & Healthcare",
        "utilities": "Energy & Utilities",
        "energy": "Energy & Utilities",
        "basic materials": "Metals & Mining",
        "consumer defensive": "FMCG & Consumer",
        "industrials": "Industrials & Cement",
    }

    matched_sector = None
    for key, name in sector_mapping.items():
        if key in yf_sector:
            matched_sector = name
            break

    if matched_sector and matched_sector in SECTOR_POOLS:
        pool = SECTOR_POOLS[matched_sector]
        peers = [s for s in pool if s["symbol"].split(".")[0].upper().strip() != clean_target]
        return matched_sector, peers

    # Fallback to NIFTY 50 top stocks
    default_peers = [s for s in NIFTY50_STOCKS if s["symbol"].split(".")[0].upper().strip() != clean_target]
    return "Market Peers", default_peers[:8]


