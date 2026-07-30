"""
alerts/telegram.py
==================
Telegram Bot alert integration.
"""

from __future__ import annotations

import logging

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_telegram_alert(message: str, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    """
    Send a message via Telegram Bot API.

    Args:
        message: The text message to send (supports HTML).
        chat_id: Target chat ID (defaults to config value).

    Returns:
        True if sent successfully, False otherwise.
    """
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram alert sent successfully.")
        return True
    except requests.RequestException as exc:
        logger.error("Telegram alert failed: %s", exc)
        return False


def format_signal_message(
    symbol: str,
    signal: str,
    confidence: float,
    entry: float,
    stop_loss: float,
    take_profit: float,
    rr: float,
) -> str:
    """
    Format a rich trade signal message for Telegram.

    Returns:
        HTML-formatted message string.
    """
    emoji_map = {
        "Strong Buy": "🚀",
        "Buy": "📈",
        "Hold": "⏸",
        "Sell": "📉",
        "Strong Sell": "🔻",
    }
    emoji = emoji_map.get(signal, "📊")

    return (
        f"{emoji} <b>StockSense AI Alert</b>\n\n"
        f"<b>Stock:</b> {symbol}\n"
        f"<b>Signal:</b> {signal} ({confidence:.1f}% confidence)\n\n"
        f"<b>Entry:</b> ₹{entry:,.2f}\n"
        f"<b>Stop Loss:</b> ₹{stop_loss:,.2f}\n"
        f"<b>Target:</b> ₹{take_profit:,.2f}\n"
        f"<b>Risk:Reward:</b> 1:{rr:.1f}\n\n"
        f"<i>This is not financial advice. Trade responsibly.</i>"
    )
