"""
tools/validate_tickers.py
=========================
Check every ticker in config.py still resolves on Yahoo Finance.

Indian tickers change more often than you would expect — companies rename
(Zomato → Eternal), demerge (Tata Motors → TMPV), or the symbol in the list is
simply a typo (INDIHOTEL vs the real INDHOTEL). A dead symbol fails quietly:
the scanner logs a warning and skips it, so a stock silently disappears from
every scan and sector pool without anyone noticing.

Run it after editing the lists, and periodically:

    python tools/validate_tickers.py

Exits non-zero if anything fails to resolve, so it can gate CI.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yfinance as yf  # noqa: E402

from config import ALL_STOCKS, SECTOR_POOLS  # noqa: E402

MAX_WORKERS = 8


def collect_symbols() -> dict[str, str]:
    """Every unique symbol across the config lists, mapped to its label."""
    symbols: dict[str, str] = {}
    for entry in ALL_STOCKS:
        symbols.setdefault(entry["symbol"], entry["name"])
    for pool in SECTOR_POOLS.values():
        for entry in pool:
            symbols.setdefault(entry["symbol"], entry["name"])
    return symbols


def check(item: tuple[str, str]) -> tuple[str, str, bool]:
    symbol, name = item
    try:
        df = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
        return symbol, name, not df.empty
    except Exception:  # noqa: BLE001 - any failure means "cannot use this symbol"
        return symbol, name, False


def main() -> int:
    symbols = collect_symbols()
    print(f"Checking {len(symbols)} unique tickers against Yahoo Finance…\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(check, sorted(symbols.items())))

    dead = [(s, n) for s, n, ok in results if not ok]

    if not dead:
        print(f"All {len(results)} tickers resolve.")
        return 0

    print(f"{len(dead)} of {len(results)} tickers did NOT resolve:\n")
    for symbol, name in dead:
        print(f"  {symbol:<20} {name}")
    print(
        "\nFix these in config.py. Confirm the replacement actually is the same "
        "company before substituting — a symbol that merely resolves is not "
        "enough (LTTS is not LTIM)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
