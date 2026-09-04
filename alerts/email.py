"""
alerts/email.py
===============
SMTP email alert integration.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import (
    EMAIL_PASSWORD,
    EMAIL_RECEIVER,
    EMAIL_SENDER,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PORT,
)

logger = logging.getLogger(__name__)


def send_email_alert(
    subject: str,
    body_html: str,
    receiver: str = "",
    sender: str = "",
    password: str = "",
) -> bool:
    """
    Send an HTML email alert via SMTP (Gmail / any SMTP).

    Credentials passed in take precedence over the configured values, so the
    sidebar fields actually do something. Previously they were collected and
    ignored, and the "Test Email" button could never succeed.

    Args:
        subject:    Email subject.
        body_html:  HTML body content.
        receiver:   Recipient address. Falls back to config.
        sender:     From address. Falls back to config.
        password:   SMTP password / app password. Falls back to config.

    Returns:
        True on success, False on failure.
    """
    from_addr = (sender or EMAIL_SENDER or "").strip()
    to_addr = (receiver or EMAIL_RECEIVER or "").strip()
    secret = password or EMAIL_PASSWORD or ""

    if not from_addr or not secret or not to_addr:
        logger.warning("Email not configured. Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.login(from_addr, secret)
            server.sendmail(from_addr, to_addr, msg.as_string())
        logger.info("Email alert sent to %s", to_addr)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Email alert failed: %s", exc)
        return False


def format_signal_email(
    symbol: str,
    signal: str,
    confidence: float,
    entry: float,
    stop_loss: float,
    take_profit: float,
    rr: float,
    reasons: list[str],
) -> tuple[str, str]:
    """
    Generate subject and HTML body for a trade signal email.

    Returns:
        Tuple of (subject, html_body).
    """
    reasons_html = "".join(f"<li>{r}</li>" for r in reasons[:8])
    subject = f"StockSense AI | {signal} Signal for {symbol}"
    body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#111;color:#eee;padding:20px;">
        <h2 style="color:#00e676;">📊 StockSense AI — Trade Alert</h2>
        <table style="width:100%;max-width:500px;border-collapse:collapse;">
            <tr><td style="padding:8px;color:#aaa;">Stock</td><td style="padding:8px;font-weight:bold;">{symbol}</td></tr>
            <tr><td style="padding:8px;color:#aaa;">Signal</td><td style="padding:8px;font-weight:bold;color:#00e676;">{signal} ({confidence:.1f}%)</td></tr>
            <tr><td style="padding:8px;color:#aaa;">Entry</td><td style="padding:8px;">₹{entry:,.2f}</td></tr>
            <tr><td style="padding:8px;color:#aaa;">Stop Loss</td><td style="padding:8px;color:#f44336;">₹{stop_loss:,.2f}</td></tr>
            <tr><td style="padding:8px;color:#aaa;">Target</td><td style="padding:8px;color:#00e676;">₹{take_profit:,.2f}</td></tr>
            <tr><td style="padding:8px;color:#aaa;">Risk:Reward</td><td style="padding:8px;">1:{rr:.1f}</td></tr>
        </table>
        <h3 style="color:#ffd740;">Analysis</h3>
        <ul style="color:#ccc;">{reasons_html}</ul>
        <p style="color:#888;font-size:12px;">This is not financial advice. Trade responsibly.</p>
    </body></html>
    """
    return subject, body
