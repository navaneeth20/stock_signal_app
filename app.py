"""
app.py
======
StockSense AI — AI-Powered Indian Stock Market Signal Dashboard
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    ALL_STOCKS,
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    AI_MAX_TOKENS,
    AI_MODEL,
    BT_DEFAULT_CAPITAL,
    DEFAULT_CAPITAL,
    DEFAULT_INTERVAL,
    DEFAULT_RISK_PER_TRADE,
    INDEX_GROUPS,
    NIFTY50_STOCKS,
    OPENAI_API_KEY,

    OPENAI_BASE_URL,
    SIGNAL_COLORS,
    SIGNAL_EMOJI,
    SUPPORTED_INTERVALS,
)
from alerts.email import format_signal_email, send_email_alert
from alerts.telegram import format_signal_message, send_telegram_alert
from backtesting.backtest import run_backtest
from charts.candlestick import build_equity_curve, build_price_chart
from data.fetch_data import (
    fetch_ohlcv,
    get_company_info,
    get_sector_peers,
    get_stock_name,
    normalise_symbol,
)
from data.news_sentiment import fetch_news_sentiment

from database import (
    add_to_watchlist,
    create_or_update_user,
    get_all_users,
    get_eod_summary,
    get_recent_signals,
    get_search_history,
    get_user_by_email,
    get_user_by_phone,
    get_watchlist,
    initialise_db,
    is_in_watchlist,
    log_search_event,
    remove_from_watchlist,
    save_signal,
)
from indicators.mtf import compute_mtf_alignment
from reports import (
    INSTITUTIONAL_PROMPTS,
    call_gemini_api,
    call_openai_api,
    generate_fallback_institutional_report,
)

from strategies.industry_engine import analyze_sector_performance
from strategies.risk import calculate_risk


from strategies.signal_engine import compute_all_indicators, generate_signal
from utils.helpers import color_for_signal, format_inr, format_volume, pct_change
from utils.quant_risk import run_monte_carlo_simulation


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/",
        "About": f"**{APP_NAME} v{APP_VERSION}** — {APP_TAGLINE}",
    },
)

# ── Global Layout & Typography CSS ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Typography & Monospace ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}
.mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Streamlit Top Header Transparent Bar ── */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background-color: transparent !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* ── Segmented Navigation Bar Pill Layout ── */
div[data-testid="stRadio"] {
    margin-bottom: 22px !important;
}
div[data-testid="stRadio"] > label {
    display: none !important;
}
div[data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}

/* Individual Pill Base Style */
div[data-testid="stRadio"] [role="radiogroup"] label {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 10px 18px !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 11.5px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin: 0 !important;
    user-select: none !important;
    line-height: 1.2 !important;
}

/* Hide Radio Circle Dots */
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child,
div[data-testid="stRadio"] [role="radiogroup"] label input[type="radio"],
div[data-testid="stRadio"] [role="radiogroup"] label span:first-child:not([data-testid="stMarkdownContainer"] *) {
    display: none !important;
    visibility: hidden !important;
    width: 0px !important;
    height: 0px !important;
}



/* ── General Header Components ── */
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(46, 160, 67, 0.12);
    border: 1px solid rgba(46, 160, 67, 0.35);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    color: #3fb950;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pulsing-dot {
    width: 7px;
    height: 7px;
    background-color: #3fb950;
    border-radius: 50%;
    box-shadow: 0 0 10px #3fb950;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(63, 185, 80, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
}

    box-shadow: 0 8px 24px rgba(46, 160, 67, 0.45) !important;
}

/* ── Inputs & Selectboxes ── */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background-color: rgba(13, 17, 23, 0.7) !important;
    border-color: rgba(88, 166, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #f0f6fc !important;
}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080b10; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

/* ── Mobile Responsive ── */
@media (max-width: 768px) {
    .hero-header {
        padding: 16px 14px !important;
        margin-bottom: 16px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        white-space: nowrap !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 7px 12px !important;
        font-size: 12px !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ────────────────────────────────────────────────────────
def _init_session() -> None:
    defaults = {
        "df": None,
        "signal_result": None,
        "selected_symbol": "RELIANCE.NS",
        "auto_refresh": False,
        "last_refresh": 0.0,
        "scanner_results": None,
        "user": None,
        "is_logged_in": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()
initialise_db()


def render_login_page() -> None:
    """Render an Institutional Enterprise Access Portal."""
    st.markdown(
        f"""
        <div style="max-width:620px; margin: 30px auto 20px auto; padding:36px; background:linear-gradient(145deg, rgba(13,17,23,0.95), rgba(22,27,34,0.98)); border:1px solid rgba(88,166,255,0.3); border-radius:20px; box-shadow:0 24px 60px rgba(0,0,0,0.7); text-align:center;">
            <div style="display:inline-flex; align-items:center; justify-content:center; width:56px; height:56px; background:linear-gradient(135deg, #1f6feb, #388bfd); border-radius:14px; box-shadow:0 6px 20px rgba(31,111,235,0.4); margin-bottom:16px;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div style="font-size:11px; font-weight:700; color:#58a6ff; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:6px;">Enterprise Institutional Access</div>
            <h2 style="color:#f0f6fc; margin:0 0 8px 0; font-weight:800; font-size:25px; letter-spacing:-0.01em;">{APP_NAME} Terminal</h2>
            <p style="color:#8b949e; font-size:13.5px; margin-bottom:0;">{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        registered_users = get_all_users()
        
        login_tab1, login_tab2 = st.tabs(["MEMBER ACCESS", "NEW ACCOUNT SETUP"])
        
        with login_tab1:
            if registered_users:
                st.markdown("<div style='font-size:13px; font-weight:700; color:#8b949e; margin-bottom:8px;'>SELECT REGISTERED INSTITUTIONAL PROFILE</div>", unsafe_allow_html=True)
                user_options = {
                    f"{u['name'].upper()} • {u['email']} (Phone: {u['phone']})": u for u in registered_users
                }
                selected_user_str = st.selectbox("Registered Accounts", list(user_options.keys()), key="select_user_dropdown")
                if st.button("ACCESS INSTITUTIONAL TERMINAL", key="btn_quick_signin", use_container_width=True, type="primary"):
                    user_data = user_options[selected_user_str]
                    updated_user = create_or_update_user(user_data['name'], user_data['phone'], user_data['email'])
                    st.session_state['user'] = updated_user
                    st.session_state['is_logged_in'] = True
                    st.success(f"Authenticated successfully as {user_data['name']}.")
                    time.sleep(0.3)
                    st.rerun()
            else:
                st.info("No saved accounts found in session audit log. Please set up your credentials below.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:13px; font-weight:700; color:#8b949e; margin-bottom:8px;'>OR SEARCH VIA EMAIL / PHONE</div>", unsafe_allow_html=True)
            lookup_query = st.text_input("Email Address or Phone Number", key="lookup_input", placeholder="e.g. trader@institution.com or +91 9876543210")
            if st.button("Authenticate Profile", key="btn_lookup_signin", use_container_width=True):
                if lookup_query.strip():
                    found_user = get_user_by_email(lookup_query) or get_user_by_phone(lookup_query)
                    if found_user:
                        updated_user = create_or_update_user(found_user['name'], found_user['phone'], found_user['email'])
                        st.session_state['user'] = updated_user
                        st.session_state['is_logged_in'] = True
                        st.success(f"Authenticated as {found_user['name']}.")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("No account record matched that email or phone number.")
                else:
                    st.warning("Please enter your registered email address or phone number.")

        with login_tab2:
            st.markdown("<div style='font-size:13px; font-weight:700; color:#8b949e; margin-bottom:8px;'>CREATE ENTERPRISE USER PROFILE</div>", unsafe_allow_html=True)
            with st.form("registration_form"):
                reg_name = st.text_input("Full Legal Name", placeholder="e.g. Navaneeth Kumar")
                reg_phone = st.text_input("Phone Number (+91)", placeholder="e.g. +91 9876543210")
                reg_email = st.text_input("Corporate Email Address", placeholder="e.g. navaneeth@firm.com")
                submit_reg = st.form_submit_button("REGISTER & LAUNCH TERMINAL", use_container_width=True)


                if submit_reg:
                    if not reg_name.strip():
                        st.error("Please enter your Full Name.")
                    elif not reg_phone.strip():
                        st.error("Please enter your Phone Number.")
                    elif not reg_email.strip() or "@" not in reg_email:
                        st.error("Please enter a valid Email Address.")
                    else:
                        user_rec = create_or_update_user(reg_name, reg_phone, reg_email)
                        st.session_state['user'] = user_rec
                        st.session_state['is_logged_in'] = True
                        st.success(f"Profile created! Welcome, {user_rec['name']}.")
                        time.sleep(0.3)
                        st.rerun()


# Enforce Login Gate
if not st.session_state.get("is_logged_in"):
    render_login_page()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo + title
    current_user = st.session_state.get("user") or {}
    user_display_name = current_user.get("name", "Trader")
    user_email = current_user.get("email", "")
    user_phone = current_user.get("phone", "")
    
    # Initials badge
    name_parts = user_display_name.strip().split()
    initials = "".join([p[0].upper() for p in name_parts[:2]]) if name_parts else "TR"

    st.markdown(
        f"""
        <div class="user-profile-card">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="background:linear-gradient(135deg,#1f6feb,#388bfd); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:16px; font-weight:800; color:#FFFFFF; box-shadow:0 4px 12px rgba(31,111,235,0.4); shrink:0;">{initials}</div>
                <div>
                    <div style="font-size:13.5px; font-weight:700; line-height:1.2;">{user_display_name}</div>
                    <div style="font-size:10px; font-weight:700; color:#58a6ff; letter-spacing:0.06em; margin-top:2px;">INSTITUTIONAL PRO</div>
                </div>
            </div>
            <div style="margin-top:10px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1); font-size:11px; opacity:0.85; word-break:break-all;">
                <div>📧 {user_email if user_email else 'N/A'}</div>
                {f'<div style="margin-top:2px;">📞 {user_phone}</div>' if user_phone else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    if st.button("Logout / Switch Profile", key="btn_logout_sidebar", use_container_width=True):
        st.session_state["is_logged_in"] = False
        st.session_state["user"] = None
        st.rerun()

    st.divider()

    # ── Theme Selector ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>INTERFACE THEME</div>", unsafe_allow_html=True)
    theme_choice = st.radio("UI Theme", ["🌙 Institutional Dark", "☀️ TailAdmin Light Dashboard"], horizontal=True, key="theme_radio")

    if theme_choice == "☀️ TailAdmin Light Dashboard":
        st.markdown(
            """
            <style>
            /* ── TailAdmin Light Theme Overrides ── */
            .stApp {
                background-color: #F8FAFC !important;
                background-image: none !important;
                color: #0F172A !important;
            }

            /* Streamlit Header Bar Transparent */
            header[data-testid="stHeader"],
            [data-testid="stHeader"] {
                background-color: transparent !important;
                background: transparent !important;
                box-shadow: none !important;
            }

            /* Sidebar Light Mode Override */
            [data-testid="stSidebar"] {
                background-color: #FFFFFF !important;
                background: #FFFFFF !important;
                border-right: 1px solid #E2E8F0 !important;
            }
            [data-testid="stSidebar"] *,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #0F172A !important;
            }
            [data-testid="stSidebar"] .section-header {
                color: #1E3A8A !important;
                border-bottom: 2px solid #E2E8F0 !important;
            }

            /* User Profile Card in Light Mode */
            .user-profile-card {
                background: #FFFFFF !important;
                border: 1px solid #CBD5E1 !important;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
                border-radius: 14px !important;
                padding: 16px 14px !important;
                margin-bottom: 14px !important;
                color: #0F172A !important;
            }
            .user-profile-card *, .user-profile-card div {
                color: #0F172A !important;
            }

            /* Hero Header Card */
            .hero-header {
                background: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
                color: #0F172A !important;
            }
            .hero-header h1 {
                color: #0F172A !important;
                background: none !important;
                -webkit-text-fill-color: #0F172A !important;
            }
            .hero-header p, .hero-header span, .hero-header div {
                color: #475569 !important;
            }

            /* Remove Red Underline from Tabs */
            div[data-baseweb="tab-highlight"], 
            div[data-baseweb="tab-border"],
            [data-testid="stTab"] div[data-baseweb="tab-highlight"],
            button[role="tab"] + div,
            .stTabs [data-baseweb="tab-highlight"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0px !important;
                width: 0px !important;
                background: transparent !important;
                background-color: transparent !important;
                border: none !important;
            }

            /* Tab Overflow Chevron Scroll Buttons in Light Mode */
            .stTabs button[aria-label="Previous tab"],
            .stTabs button[aria-label="Next tab"],
            div[data-baseweb="tab-list"] > button {
                background-color: #FFFFFF !important;
                background: #FFFFFF !important;
                color: #2563EB !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 8px !important;
            }

            /* Navigation Pills in Light Mode */
            div[data-testid="stRadio"] [role="radiogroup"] label {
                background: #F1F5F9 !important;
                color: #475569 !important;
                border: 1px solid #CBD5E1 !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] label p,
            div[data-testid="stRadio"] [role="radiogroup"] label span {
                color: #475569 !important;
                font-weight: 700 !important;
                font-size: 11.5px !important;
                margin: 0 !important;
                white-space: nowrap !important;
            }

            div[data-testid="stRadio"] [role="radiogroup"] label:hover {
                background: #E2E8F0 !important;
                border-color: #2563EB !important;
                transform: translateY(-1px) !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] label:hover p,
            div[data-testid="stRadio"] [role="radiogroup"] label:hover span {
                color: #1E3A8A !important;
            }

            div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
                border-color: #1D4ED8 !important;
                box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
            div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span {
                color: #FFFFFF !important;
            }

            /* Section Headers in Light Mode */
            .section-header {
                color: #1E3A8A !important;
                border-bottom: 2px solid #E2E8F0 !important;
                font-weight: 800 !important;
            }

            /* Cards & Content Containers in Light Mode */
            .placeholder-card, .hero-signal-card, .metric-card, .ai-box {
                background: #FFFFFF !important;
                border: 1px solid #E2E8F0 !important;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04) !important;
                color: #0F172A !important;
                transition: all 0.25s ease !important;
            }
            .placeholder-card:hover, .hero-signal-card:hover, .metric-card:hover, .ai-box:hover {
                border-color: #2563EB !important;
                box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12) !important;
                transform: translateY(-2px) !important;
            }
            .placeholder-card *, .hero-signal-card *, .metric-card *, .ai-box * {
                color: #0F172A !important;
            }
            .metric-label {
                color: #64748B !important;
                font-weight: 700 !important;
            }

            /* Streamlit Buttons in Light Mode */
            .stButton > button {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: 1px solid #CBD5E1 !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
                font-weight: 700 !important;
            }
            .stButton > button:hover {
                background-color: #F1F5F9 !important;
                color: #1E3A8A !important;
                border-color: #2563EB !important;
                transform: translateY(-1px) !important;
            }
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
                color: #FFFFFF !important;
                border: 1px solid #1D4ED8 !important;
                box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            }
            .stButton > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
                box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
            }
            .stButton > button[kind="primary"] * {
                color: #FFFFFF !important;
            }

            /* Streamlit Alert Boxes in Light Mode */
            div[data-testid="stAlert"] {
                background-color: #EFF6FF !important;
                border: 1px solid #BFDBFE !important;
                color: #1E40AF !important;
                border-radius: 12px !important;
            }
            div[data-testid="stAlert"] * {
                color: #1E40AF !important;
            }

            /* Inputs & Selectboxes in Light Mode */
            div[data-baseweb="select"] > div, 
            div[data-baseweb="input"] > div,
            div[data-baseweb="base-input"] {
                background-color: #FFFFFF !important;
                border-color: #CBD5E1 !important;
                color: #0F172A !important;
            }
            div[data-baseweb="select"] span,
            div[data-baseweb="input"] input,
            input, select, textarea {
                color: #0F172A !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            /* ── Institutional Dark Theme Restoration ── */
            .stApp {
                background: #090d14 !important;
                background-image: 
                    radial-gradient(circle at 50% 0%, rgba(31, 111, 235, 0.12) 0%, transparent 60%),
                    radial-gradient(rgba(56, 139, 253, 0.04) 1px, transparent 0) !important;
                background-size: 100% 100%, 28px 28px !important;
                color: #f0f6fc !important;
            }

            /* Streamlit Header Bar Transparent */
            header[data-testid="stHeader"],
            [data-testid="stHeader"] {
                background-color: transparent !important;
                background: transparent !important;
                box-shadow: none !important;
            }

            /* Sidebar Dark Mode */
            [data-testid="stSidebar"] {
                background-color: #0b0e14 !important;
                background: #0b0e14 !important;
                border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            }
            [data-testid="stSidebar"] *,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #c9d1d9 !important;
            }
            [data-testid="stSidebar"] .section-header {
                color: #58a6ff !important;
                border-bottom: 1px solid rgba(88, 166, 255, 0.15) !important;
            }

            /* User Profile Card in Dark Mode */
            .user-profile-card {
                background: linear-gradient(145deg, rgba(22,27,34,0.9), rgba(13,17,23,0.95)) !important;
                border: 1px solid rgba(88,166,255,0.2) !important;
                box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
                border-radius: 14px !important;
                padding: 16px 14px !important;
                margin-bottom: 14px !important;
                color: #f0f6fc !important;
            }
            .user-profile-card *, .user-profile-card div {
                color: #f0f6fc !important;
            }

            /* Hero Header Card */
            .hero-header {
                background: linear-gradient(135deg, rgba(13, 17, 23, 0.95) 0%, rgba(22, 27, 34, 0.85) 100%) !important;
                border: 1px solid rgba(88, 166, 255, 0.2) !important;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6) !important;
                color: #f0f6fc !important;
            }
            .hero-header h1 {
                color: #f0f6fc !important;
                background: linear-gradient(90deg, #58a6ff, #00e5ff, #79c0ff) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }
            .hero-header p, .hero-header span, .hero-header div {
                color: #8b949e !important;
            }

            /* Navigation Pills in Dark Mode */
            div[data-testid="stRadio"] [role="radiogroup"] label {
                background: rgba(22, 27, 34, 0.8) !important;
                color: #8b949e !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] label p,
            div[data-testid="stRadio"] [role="radiogroup"] label span {
                color: #8b949e !important;
                font-weight: 700 !important;
                font-size: 11.5px !important;
                margin: 0 !important;
                white-space: nowrap !important;
            }

            div[data-testid="stRadio"] [role="radiogroup"] label:hover {
                background: rgba(31, 111, 235, 0.25) !important;
                border-color: rgba(88, 166, 255, 0.4) !important;
                transform: translateY(-1px) !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] label:hover p,
            div[data-testid="stRadio"] [role="radiogroup"] label:hover span {
                color: #58a6ff !important;
            }

            div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
                border-color: #388bfd !important;
                box-shadow: 0 4px 16px rgba(31, 111, 235, 0.45) !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
            div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span {
                color: #FFFFFF !important;
            }



            /* Section Headers in Dark Mode */
            .section-header {
                color: #58a6ff !important;
                border-bottom: 1px solid rgba(88, 166, 255, 0.15) !important;
                font-weight: 700 !important;
            }

            /* Cards & Content Containers in Dark Mode */
            .placeholder-card, .hero-signal-card, .metric-card, .ai-box {
                background: linear-gradient(145deg, rgba(13, 20, 32, 0.9), rgba(9, 13, 20, 0.98)) !important;
                border: 1px solid rgba(88, 166, 255, 0.2) !important;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5) !important;
                color: #f0f6fc !important;
                transition: all 0.25s ease !important;
            }
            .placeholder-card:hover, .hero-signal-card:hover, .metric-card:hover, .ai-box:hover {
                border-color: rgba(88, 166, 255, 0.45) !important;
                box-shadow: 0 18px 48px rgba(0, 0, 0, 0.6), 0 0 18px rgba(88, 166, 255, 0.18) !important;
                transform: translateY(-2px) !important;
            }
            .placeholder-card *, .hero-signal-card *, .metric-card *, .ai-box * {
                color: #f0f6fc !important;
            }
            .metric-label {
                color: #8b949e !important;
                font-weight: 700 !important;
            }

            /* Buttons in Dark Mode */
            .stButton > button {
                background-color: rgba(22, 27, 34, 0.8) !important;
                color: #c9d1d9 !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
                font-weight: 600 !important;
            }
            .stButton > button:hover {
                background-color: rgba(56, 139, 253, 0.2) !important;
                color: #58a6ff !important;
                border-color: rgba(88, 166, 255, 0.4) !important;
                transform: translateY(-1px) !important;
            }
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
                color: #ffffff !important;
                border: 1px solid #3fb950 !important;
                box-shadow: 0 4px 16px rgba(46, 160, 67, 0.3) !important;
            }
            .stButton > button[kind="primary"] * {
                color: #ffffff !important;
            }

            /* Inputs & Selectboxes in Dark Mode */
            div[data-baseweb="select"] > div, 
            div[data-baseweb="input"] > div,
            div[data-baseweb="base-input"] {
                background-color: rgba(13, 17, 23, 0.7) !important;
                border-color: rgba(88, 166, 255, 0.2) !important;
                color: #f0f6fc !important;
            }
            div[data-baseweb="select"] span,
            div[data-baseweb="input"] input,
            input, select, textarea {
                color: #f0f6fc !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )






    st.divider()



    # ── Stock Selection ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>STOCK SELECTION</div>", unsafe_allow_html=True)

    # Build symbol→name mapping for display
    all_stocks = ALL_STOCKS
    symbol_map = {s["symbol"]: f"{s['name']} ({s['symbol']})" for s in all_stocks}


    search_input = st.text_input(
        "Custom Symbol",
        placeholder="e.g. WIPRO, TATAMOTORS",
        help="Enter any NSE symbol without .NS suffix",
        key="custom_symbol_input",
    )
    if search_input.strip():
        custom_sym = normalise_symbol(search_input.strip(), "NSE")
        selected_symbol = custom_sym
    else:
        selected_display = st.selectbox(
            "Select Stock",
            options=list(symbol_map.keys()),
            format_func=lambda s: symbol_map[s],
            index=0,
        )
        selected_symbol = selected_display

    exchange = st.selectbox("Exchange", ["NSE", "BSE"])

    st.divider()

    # ── Timeframe ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>TIMEFRAME & PERIOD</div>", unsafe_allow_html=True)
    interval_label = st.selectbox("Interval", list(SUPPORTED_INTERVALS.keys()), index=0)
    interval = SUPPORTED_INTERVALS[interval_label]

    use_custom_dates = st.checkbox("Custom Date Range", value=False)
    if use_custom_dates:
        col_d1, col_d2 = st.columns(2)
        start_date = col_d1.date_input("From", datetime.now() - timedelta(days=365))
        end_date = col_d2.date_input("To", datetime.now())
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
    else:
        period_label = st.selectbox(
            "Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
            index=3,
        )
        period_map = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y",
        }
        period = period_map[period_label]
        start_dt = None
        end_dt = None

    st.divider()

    # ── Indicators Toggle ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>TECHNICAL OVERLAYS</div>", unsafe_allow_html=True)
    show_ema = st.checkbox("EMA (20/50/200)", value=True)
    show_supertrend = st.checkbox("Supertrend", value=True)
    show_bollinger = st.checkbox("Bollinger Bands", value=True)
    show_vwap = st.checkbox("VWAP", value=True)
    show_volume = st.checkbox("Volume", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)

    st.divider()

    # ── Risk Settings ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>RISK PARAMETERS</div>", unsafe_allow_html=True)
    capital = st.number_input(
        "Capital (₹)",
        min_value=10_000,
        max_value=10_000_000,
        value=DEFAULT_CAPITAL,
        step=10_000,
        format="%d",
    )
    risk_pct = st.slider("Risk Per Trade (%)", 0.5, 5.0, DEFAULT_RISK_PER_TRADE * 100, 0.5)

    st.divider()

    # ── Auto Refresh ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>AUTO REFRESH ENGINE</div>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Enable Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh every (seconds)", 30, 600, 60, 30)

    # ── Load Button ──────────────────────────────────────────────────────────
    st.divider()
    load_clicked = st.button("RUN QUANT ANALYSIS", type="primary", use_container_width=True)

    # ── Alerts Config ────────────────────────────────────────────────────────
    with st.expander("Alert Channels & Notifications"):

        tg_token = st.text_input("Telegram Bot Token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
        tg_chat = st.text_input("Telegram Chat ID", value=os.getenv("TELEGRAM_CHAT_ID", ""))
        email_from = st.text_input("Email (From)", value=os.getenv("EMAIL_SENDER", ""))
        email_pass = st.text_input("Email Password", type="password")
        email_to = st.text_input("Email (To)", value=os.getenv("EMAIL_RECEIVER", ""))

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Load & Analyse
# ═══════════════════════════════════════════════════════════════════════════════

def _find_better_alternatives(current_symbol: str, current_confidence: float) -> tuple[str, list[dict]]:
    """Scan sector/category peer stocks for higher confidence or stronger signals."""
    sector_name, peer_pool = get_sector_peers(current_symbol)
    better_list = []
    for item in peer_pool:
        sym = item["symbol"]
        if sym.upper() == current_symbol.upper():
            continue
        try:
            raw = fetch_ohlcv(sym, period="1y", interval="1d")
            if raw.empty or len(raw) < 60:
                continue
            d = compute_all_indicators(raw)
            res = generate_signal(sym, d)
            if res.confidence >= current_confidence or (res.signal in ("Strong Buy", "Buy") and res.confidence >= 60):
                last_p = d["Close"].iloc[-1]
                prev_p = d["Close"].iloc[-2] if len(d) > 1 else last_p
                chg = pct_change(float(prev_p), float(last_p))
                better_list.append({
                    "symbol": sym,
                    "name": item["name"],
                    "signal": res.signal,
                    "confidence": res.confidence,
                    "price": last_p,
                    "change": chg,
                    "rr": res.risk_reward,
                    "sector": sector_name,
                })
        except Exception:
            continue
    better_list.sort(key=lambda x: x["confidence"], reverse=True)
    return sector_name, better_list[:3]


def load_and_analyse(symbol: str) -> None:
    """Fetch data and run full signal analysis, store in session state."""
    with st.spinner(f"Fetching & Analysing {symbol}…"):
        try:
            if use_custom_dates:
                df_raw = fetch_ohlcv(symbol, interval=interval, start=start_dt, end=end_dt)
            else:
                df_raw = fetch_ohlcv(symbol, interval=interval, period=period)

            df = compute_all_indicators(df_raw)
            result = generate_signal(symbol, df)
            risk = calculate_risk(df, result.signal, capital=capital, risk_per_trade=risk_pct / 100)

            # Get full official company name
            comp_name = get_stock_name(symbol)

            # 1. Multi-Timeframe (MTF) Alignment Analysis
            mtf_res = compute_mtf_alignment(symbol)
            result.mtf_result = mtf_res
            # Apply MTF confidence modifier
            result.confidence = max(0.0, min(100.0, result.confidence + mtf_res.confidence_modifier))

            # 2. Market Sentiment & News Intelligence
            news_res = fetch_news_sentiment(symbol, comp_name)
            result.news_result = news_res

            # 3. Quantitative Risk & Monte Carlo Simulation
            mc_res = run_monte_carlo_simulation(df, risk.entry_price, risk.stop_loss, risk.take_profit)
            result.mc_result = mc_res

            # Find better sector/category alternatives
            sector_category, alternatives = _find_better_alternatives(symbol, result.confidence)

            # Persist signal & search event
            save_signal(
                symbol=symbol,
                signal=result.signal,
                confidence=result.confidence,
                entry_price=risk.entry_price,
                stop_loss=risk.stop_loss,
                take_profit=risk.take_profit,
                risk_reward=risk.risk_reward,
                interval=interval,
                reasons=result.reasons,
                mtf_status=mtf_res.alignment_status if mtf_res else "",
                win_prob=mc_res.win_probability if mc_res else 0.0,
                source="Signal Terminal",
            )

            log_search_event(
                symbol=symbol,
                name=comp_name,
                signal=result.signal,
                confidence=result.confidence,
                price=risk.entry_price,
                source="Signal Terminal",
            )



            st.session_state.df = df
            st.session_state.signal_result = result
            st.session_state.risk = risk
            st.session_state.company_name = comp_name
            st.session_state.sector_category = sector_category
            st.session_state.better_alternatives = alternatives
            st.session_state.selected_symbol = symbol
            st.session_state.last_refresh = time.time()
            logger.info("Analysis complete for %s (%s): %s", symbol, comp_name, result.signal)

        except Exception as exc:
            st.error(f"❌ Error loading {symbol}: {exc}")
            logger.exception("load_and_analyse failed for %s", symbol)




# Auto-refresh logic
if auto_refresh:
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed > refresh_interval:
        load_and_analyse(selected_symbol)

# Trigger on button click
if load_clicked:
    load_and_analyse(selected_symbol)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="hero-header">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
            <div style="display:flex; align-items:center; gap:18px;">
                <div style="background:linear-gradient(135deg,#1f6feb,#58a6ff); width:52px; height:52px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:26px; box-shadow:0 8px 24px rgba(31,111,235,0.4);">📈</div>
                <div>
                    <div style="display:flex; align-items:center; gap:12px;">
                        <h1 style="margin:0; font-size:28px; font-weight:800; background:linear-gradient(90deg,#58a6ff,#00e5ff,#79c0ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.02em;">{APP_NAME} <span style="font-size:13px; font-weight:700; padding:3px 10px; border-radius:20px; background:rgba(88,166,255,0.12); color:#58a6ff; border:1px solid rgba(88,166,255,0.25); -webkit-text-fill-color:initial;">PRO TERMINAL</span></h1>
                    </div>
                    <p style="margin:4px 0 0; color:#8b949e; font-size:13.5px; font-weight:500;">{APP_TAGLINE}</p>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:14px;">
                <div class="live-indicator"><span class="pulsing-dot"></span> LIVE NSE / BSE</div>
                <div class="mono-font" style="font-size:12px; color:#8b949e; background:rgba(255,255,255,0.04); padding:6px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">v{APP_VERSION}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — AI Explanation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_ai_explanation(result, risk) -> str:
    """
    Generate a natural-language trade explanation.
    Uses OpenAI API if configured, otherwise generates a rule-based explanation.
    """
    sig_emoji = SIGNAL_EMOJI.get(result.signal, "📊")

    # Try OpenAI API
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            prompt = (
                f"You are a professional Indian stock market analyst. "
                f"Generate a concise 5-7 sentence trading explanation for the following signal:\n\n"
                f"Stock: {result.symbol}\n"
                f"Signal: {result.signal} ({result.confidence:.1f}% confidence)\n"
                f"Entry: ₹{risk.entry_price:.2f}, Stop Loss: ₹{risk.stop_loss:.2f}, "
                f"Target: ₹{risk.take_profit:.2f}, RR: 1:{risk.risk_reward:.1f}\n"
                f"Key reasons: {'; '.join(result.reasons[:5])}\n\n"
                "Be professional, concise, and mention risk management."
            )
            resp = client.chat.completions.create(
                model=AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=AI_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("OpenAI call failed: %s", exc)

    # Rule-based fallback
    trend = (
        "bullish" if result.signal in ("Strong Buy", "Buy")
        else "bearish" if result.signal in ("Sell", "Strong Sell")
        else "neutral"
    )
    lines = [
        f"<b>{sig_emoji} {result.signal} — {result.confidence:.1f}% Confidence</b><br><br>",
        f"The overall trend for <b>{result.symbol}</b> is currently <b>{trend}</b>. ",
    ]
    for r in result.reasons[:4]:
        lines.append(f"{r}. ")
    lines.append(
        f"<br><br><b>Risk Management:</b> Entry at <b>₹{risk.entry_price:,.2f}</b>, "
        f"stop loss at <b>₹{risk.stop_loss:,.2f}</b> ({risk.stop_pct:.1f}% below entry), "
        f"and target at <b>₹{risk.take_profit:,.2f}</b> gives a risk-to-reward ratio of "
        f"<b>1:{risk.risk_reward:.1f}</b>. "
    )
    lines.append(
        "<br><br><i>⚠️ This analysis is for educational purposes only. "
        "Always manage your risk and trade responsibly.</i>"
    )
    return "".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SEGMENTED NAVIGATION BAR
# ═══════════════════════════════════════════════════════════════════════════════

active_tab = st.radio(
    "Navigation Menu",
    [
        "SIGNAL TERMINAL",
        "TECHNICAL ANALYSIS",
        "QUANT BACKTEST",
        "MARKET SCANNER",
        "SECTOR & INDUSTRY PERFORMANCE",
        "WATCHLIST & AUDIT LOGS",
        "INSTITUTIONAL RESEARCH",
        "ALERT MANAGER",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_segmented_nav",
)




# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SIGNAL TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

if active_tab == "SIGNAL TERMINAL":

    if st.session_state.signal_result is None:
        st.markdown(
            """
            <div class="placeholder-card" style="text-align:center; padding:70px 20px; max-width:650px; margin:30px auto; border-radius:18px;">
                <div style="display:inline-flex; align-items:center; justify-content:center; width:64px; height:64px; background:rgba(56,139,253,0.12); border:1px solid rgba(56,139,253,0.3); border-radius:16px; margin-bottom:18px;">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#58A6FF" stroke-width="2"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
                </div>
                <h2 style="font-size:22px; font-weight:800; margin:0 0 8px 0; letter-spacing:-0.01em;">QUANTITATIVE SIGNAL TERMINAL</h2>
                <p style="font-size:13.5px; line-height:1.6; margin:0;">
                    Select an Indian equity ticker from the left sidebar and click <strong>RUN QUANT ANALYSIS</strong> to execute multi-timeframe indicator computation, signal scoring, and risk management scenarios.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    else:
        result = st.session_state.signal_result
        risk = st.session_state.risk
        df = st.session_state.df
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        sig_color = SIGNAL_COLORS.get(result.signal, "#9e9e9e")
        sig_emoji = SIGNAL_EMOJI.get(result.signal, "📊")
        price_change = pct_change(float(prev["Close"]), float(last["Close"]))

        comp_display = st.session_state.get("company_name", result.symbol)
        # ── Top Signal Card ─────────────────────────────────────────────────
        # ── Top Signal Card ─────────────────────────────────────────────────
        change_bg = 'rgba(0,230,118,0.15)' if price_change >= 0 else 'rgba(255,23,68,0.15)'
        change_fg = '#00e676' if price_change >= 0 else '#ff1744'
        change_border = 'rgba(0,230,118,0.3)' if price_change >= 0 else 'rgba(255,23,68,0.3)'
        arrow = '▲' if price_change >= 0 else '▼'

        st.markdown(
            f"""<div class="hero-signal-card" style="border: 1px solid {sig_color}55; box-shadow: 0 20px 60px -10px {sig_color}25;">
<div style="display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,0.05); padding:6px 16px; border-radius:20px; border:1px solid rgba(255,255,255,0.1); margin-bottom:12px; flex-wrap:wrap; justify-content:center;">
<span style="font-size:18px;">{sig_emoji}</span>
<span class="mono-font" style="font-size:13px; font-weight:700; color:#8b949e; letter-spacing:0.08em; text-transform:uppercase;">{result.symbol} • EQUITIES</span>
<span class="mono-font" style="font-size:12px; font-weight:700; color:#58a6ff; background:rgba(88,166,255,0.12); padding:3px 10px; border-radius:12px; border:1px solid rgba(88,166,255,0.25);">⏳ Signal Active: {result.signal_age_days} Days</span>
</div>
<div style="font-size:24px; font-weight:800; color:#f0f6fc; margin-bottom:8px;">{comp_display}</div>
<div style="font-size:42px; font-weight:800; color:{sig_color}; text-shadow:0 0 35px {sig_color}88; margin-bottom:10px; letter-spacing:-0.02em;">{result.signal}</div>
<div class="mono-font" style="font-size:22px; color:#f0f6fc; font-weight:700; margin-bottom:20px;">₹{last['Close']:,.2f} <span style="font-size:15px; font-weight:600; padding:3px 10px; border-radius:8px; margin-left:8px; background:{change_bg}; color:{change_fg}; border:1px solid {change_border}">{arrow} {abs(price_change):.2f}%</span></div>
<div style="font-size:12px; font-weight:600; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">Signal Confidence Score</div>
<div style="background:rgba(0,0,0,0.4); border-radius:10px; height:10px; width:65%; margin:0 auto 14px; padding:2px; border:1px solid rgba(255,255,255,0.06);"><div style="background:linear-gradient(90deg, {sig_color}bb, {sig_color}); height:100%; border-radius:8px; width:{result.confidence}%; box-shadow:0 0 12px {sig_color}aa;"></div></div>
<div class="mono-font" style="font-size:30px; font-weight:800; color:{sig_color}; text-shadow:0 0 20px {sig_color}66;">{result.confidence:.1f}%</div>
</div>""",
            unsafe_allow_html=True,
        )


        # ── Key Metrics Row ──────────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

        metrics = [
            (m1, "Entry Price", f"₹{risk.entry_price:,.2f}", "#58a6ff"),
            (m2, "Stop Loss", f"₹{risk.stop_loss:,.2f}", "#ff1744"),
            (m3, "Take Profit", f"₹{risk.take_profit:,.2f}", "#00e676"),
            (m4, "Risk:Reward", f"1:{risk.risk_reward:.1f}", "#ffb300"),
            (m5, "Active Days", f"{result.signal_age_days} Days", "#00e5ff"),
            (m6, "Horizon", f"{result.recommended_horizon}", "#ba68c8"),
            (m7, "Max Position", f"{risk.max_position_size:,} shares", "#79c0ff"),
        ]

        for col, label, value, color in metrics:
            col.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        st.markdown("<br>", unsafe_allow_html=True)

        # ── Price Performance & Peak Risk Assessment ─────────────────────────
        st.markdown('<div class="section-header">PRICE PERFORMANCE & PEAK RISK ASSESSMENT</div>', unsafe_allow_html=True)

        pct_1w = getattr(result, "pct_1w", 0.0)
        pct_2w = getattr(result, "pct_2w", 0.0)
        pct_1m = getattr(result, "pct_1m", 0.0)
        dist_52w_high = getattr(result, "dist_52w_high", 0.0)
        is_extended = getattr(result, "is_extended", False)
        extended_warning = getattr(result, "extended_warning", "")

        p1, p2, p3, p4, p5 = st.columns(5)

        color_1w = "#00e676" if pct_1w >= 0 else "#ff1744"
        color_2w = "#00e676" if pct_2w >= 0 else "#ff1744"
        color_1m = "#00e676" if pct_1m >= 0 else "#ff1744"
        color_52w = "#ffb300" if dist_52w_high <= 3.0 else "#58a6ff"

        perf_metrics = [
            (p1, "1-Day Change", f"{price_change:+.2f}%", change_fg),
            (p2, "1-Week Change (5D)", f"{pct_1w:+.2f}%", color_1w),
            (p3, "2-Week Change (10D)", f"{pct_2w:+.2f}%", color_2w),
            (p4, "1-Month Change (21D)", f"{pct_1m:+.2f}%", color_1m),
            (p5, "Below 52W High", f"{dist_52w_high:.1f}%", color_52w),
        ]

        for col, label, value, color in perf_metrics:
            col.markdown(
                f"""
                <div class="metric-card" style="border-top: 3px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if extended_warning:
            alert_color = "#ff1744" if is_extended else "#ffb300"
            alert_border = "rgba(255,23,68,0.4)" if is_extended else "rgba(255,179,0,0.4)"
            alert_bg = "rgba(255,23,68,0.12)" if is_extended else "rgba(255,179,0,0.12)"
            st.markdown(
                f"""
                <div style="background:{alert_bg}; border:1px solid {alert_border}; border-radius:12px; padding:12px 18px; margin-top:12px; margin-bottom:18px;">
                    <div style="font-size:13px; font-weight:600; color:{alert_color};">
                        <strong>⚠️ Peak Risk Warning:</strong> {extended_warning}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)



        # ── MTF Alignment Matrix Card ──────────────────────────────────────
        if hasattr(result, "mtf_result") and result.mtf_result:
            mtf = result.mtf_result
            st.markdown('<div class="section-header">⏱️ Multi-Timeframe (MTF) Alignment Matrix</div>', unsafe_allow_html=True)
            tf_cols = st.columns(3)
            tf_data = [
                (tf_cols[0], "1W Macro Trend", mtf.trends.get("1W")),
                (tf_cols[1], "1D Setup Trend", mtf.trends.get("1D")),
                (tf_cols[2], "1H Micro Entry", mtf.trends.get("1H")),
            ]
            for col, title, item in tf_data:
                if item:
                    color = "#00e676" if item.trend == "Bullish" else "#ff1744" if item.trend == "Bearish" else "#8b949e"
                    col.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 3px solid {color}; text-align:center;">
                            <div class="metric-label">{title}</div>
                            <div class="mono-font" style="font-size:18px; font-weight:700; color:{color}; margin-top:4px;">
                                {item.trend.upper()}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f"""
                <div style="background:rgba(22,27,34,0.6); border:1px solid rgba(88,166,255,0.2); border-radius:12px; padding:12px 18px; margin-top:12px; font-size:13px; color:#c9d1d9; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div><strong>Confluence Status:</strong> <span style="color:#58a6ff; font-weight:700;">{mtf.alignment_status}</span></div>
                    <div style="color:#8b949e;">{mtf.description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Two columns: Indicator scores + Analysis ──────────────────────
        col_l, col_r = st.columns([1, 1.4])

        with col_l:
            st.markdown('<div class="section-header">📊 Indicator Breakdown</div>', unsafe_allow_html=True)

            # Indicator score bars
            indicator_names = {
                "ema": "EMA Trend",
                "rsi": "RSI Momentum",
                "macd": "MACD Signal",
                "supertrend": "Supertrend",
                "adx": "ADX Strength",
                "volume": "Volume",
                "vwap": "VWAP Position",
            }
            weights = {"ema": 20, "rsi": 15, "macd": 20, "supertrend": 20, "adx": 10, "volume": 10, "vwap": 5}

            for key, label in indicator_names.items():
                raw = result.indicator_scores.get(key, 0)
                weight = weights.get(key, 10)
                pct = (raw / weight * 100) if weight > 0 else 50
                pct = max(0, min(100, pct))
                bar_color = "#00e676" if pct >= 60 else "#ffd740" if pct >= 40 else "#f44336"

                st.markdown(
                    f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:12px;color:#8b949e;margin-bottom:4px;">
                            <span>{label}</span>
                            <span style="color:{bar_color}">{pct:.0f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;">
                            <div style="background:{bar_color};width:{pct}%;height:6px;border-radius:4px;
                                        transition:width 0.4s ease;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Last bar values
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📌 Last Bar Summary</div>', unsafe_allow_html=True)
            last_row = df.iloc[-1]
            summary_items = []
            for col_name, fmt in [
                ("RSI", "{:.1f}"),
                ("ADX", "{:.1f}"),
                ("EMA_20", "₹{:.2f}"),
                ("EMA_50", "₹{:.2f}"),
                ("ATR", "₹{:.2f}"),
                ("VWAP", "₹{:.2f}"),
            ]:
                if col_name in df.columns:
                    val = last_row[col_name]
                    if pd.notna(val):
                        summary_items.append((col_name.replace("_", " "), fmt.format(val)))

            for name, val in summary_items:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;
                                padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);
                                font-size:13px;">
                        <span style="color:#8b949e;">{name}</span>
                        <span style="color:#e6edf3;font-weight:500;">{val}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_r:
            st.markdown('<div class="section-header">🤖 AI Trade Explanation</div>', unsafe_allow_html=True)

            # Generate AI explanation (rule-based fallback + optional LLM)
            ai_text = _generate_ai_explanation(result, risk)
            st.markdown(
                f'<div class="ai-box">{ai_text}</div>',
                unsafe_allow_html=True,
            )

            # News Sentiment Section
            if hasattr(result, "news_result") and result.news_result:
                news = result.news_result
                s_color = "#00e676" if "Bullish" in news.sentiment_label else "#ff1744" if "Bearish" in news.sentiment_label else "#ffb300"
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-header">📰 News Sentiment Intelligence (<span style="color:{s_color}">{news.sentiment_label}</span>)</div>', unsafe_allow_html=True)
                for art in news.articles[:3]:
                    art_color = "#00e676" if art.sentiment_label == "Bullish" else "#ff1744" if art.sentiment_label == "Bearish" else "#8b949e"
                    st.markdown(
                        f"""
                        <div class="reason-card" style="border-left-color:{art_color}; font-size:12.5px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                                <a href="{art.link}" target="_blank" style="color:#e6edf3; text-decoration:none; font-weight:600;">{art.title[:75]}…</a>
                                <span style="color:{art_color}; font-weight:700; font-size:10px; padding:1px 6px; border-radius:6px; background:{art_color}22;">{art.sentiment_label}</span>
                            </div>
                            <div style="font-size:10px; color:#8b949e;">{art.published}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Signal Reasons</div>', unsafe_allow_html=True)
            for reason in result.reasons[:10]:
                st.markdown(
                    f'<div class="reason-card">• {reason}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk Card & Monte Carlo Simulation ───────────────────────────────
        st.markdown('<div class="section-header">🛡️ Risk & Monte Carlo Quantitative Simulation (1,000 Iterations)</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        risk_metrics = [
            (rc1, "Capital Allocation", format_inr(risk.capital_allocation)),
            (rc2, "Risk Amount", format_inr(risk.risk_amount)),
            (rc3, "Stop Distance", f"{risk.stop_pct:.2f}%"),
            (rc4, "ATR Stop", format_inr(risk.atr_stop)),
        ]
        for col, label, val in risk_metrics:
            col.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:18px;">{val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if hasattr(result, "mc_result") and result.mc_result:
            mc = result.mc_result
            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc_color = "#00e676" if mc.win_probability >= 55 else "#ff1744" if mc.win_probability <= 45 else "#ffb300"
            mc_items = [
                (mc1, "Win Probability (PoP)", f"{mc.win_probability:.1f}%", mc_color),
                (mc2, "Expected Value (EV)", f"₹{mc.expected_value:,.2f}", "#58a6ff"),
                (mc3, "Value at Risk (95%)", f"₹{mc.var_95:,.2f}", "#ff1744"),
                (mc4, "20-Day Target (P50)", f"₹{mc.median_price:,.2f}", "#00e5ff"),
            ]
            for col, label, val, c in mc_items:
                col.markdown(
                    f"""
                    <div class="metric-card" style="border-top: 3px solid {c};">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="font-size:18px; color:{c};">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)


        # ── Better Stock Alternatives ──────────────────────────────────────────
        if "better_alternatives" in st.session_state and st.session_state.better_alternatives:
            sec_title = st.session_state.get("sector_category", "Sector")
            st.markdown(f'<div class="section-header">🌟 Better {sec_title} Alternatives (Industry Peers)</div>', unsafe_allow_html=True)
            alt_cols = st.columns(len(st.session_state.better_alternatives))

            for idx, (col, alt) in enumerate(zip(alt_cols, st.session_state.better_alternatives)):
                alt_sig_color = SIGNAL_COLORS.get(alt["signal"], "#58a6ff")
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 3px solid {alt_sig_color}; text-align:left; padding:16px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span class="mono-font" style="font-size:14px; font-weight:700; color:#f0f6fc;">{alt['symbol']}</span>
                                <span style="font-size:11px; font-weight:700; color:{alt_sig_color}; background:{alt_sig_color}22; padding:2px 8px; border-radius:10px; border:1px solid {alt_sig_color}44;">{alt['signal']}</span>
                            </div>
                            <div style="font-size:11px; color:#8b949e; margin-bottom:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{alt['name']}</div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:12px;">
                                <span style="color:#8b949e;">Confidence:</span>
                                <span class="mono-font" style="color:{alt_sig_color}; font-weight:700;">{alt['confidence']:.1f}%</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:12px;">
                                <span style="color:#8b949e;">Price:</span>
                                <span class="mono-font" style="color:#f0f6fc; font-weight:700;">₹{alt['price']:,.2f}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:12px; font-size:12px;">
                                <span style="color:#8b949e;">Risk:Reward:</span>
                                <span class="mono-font" style="color:#00e676; font-weight:700;">1:{alt['rr']:.1f}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(f"⚡ Analyze {alt['symbol'].replace('.NS','')}", key=f"btn_alt_{alt['symbol']}_{idx}", use_container_width=True):
                        st.session_state.selected_symbol = alt["symbol"]
                        load_and_analyse(alt["symbol"])
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)


        # ── Watchlist + Alert buttons ────────────────────────────────────────
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            in_wl = is_in_watchlist(selected_symbol)
            wl_label = "⭐ Remove from Watchlist" if in_wl else "⭐ Add to Watchlist"
            if st.button(wl_label, use_container_width=True):
                if in_wl:
                    remove_from_watchlist(selected_symbol)
                    st.success(f"Removed {selected_symbol} from watchlist")
                else:
                    add_to_watchlist(selected_symbol, exchange=exchange)
                    st.success(f"Added {selected_symbol} to watchlist")
                st.rerun()

        with btn_c2:
            if st.button("📱 Send Telegram Alert", use_container_width=True):
                msg = format_signal_message(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                )
                ok = send_telegram_alert(msg, chat_id=tg_chat or None)
                st.success("Telegram sent!") if ok else st.error("Telegram not configured or failed.")

        with btn_c3:
            if st.button("📧 Send Email Alert", use_container_width=True):
                subj, body = format_signal_email(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                    result.reasons,
                )
                ok = send_email_alert(subj, body, receiver=email_to or None)
                st.success("Email sent!") if ok else st.error("Email not configured or failed.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHART
# ═══════════════════════════════════════════════════════════════════════════════

elif active_tab == "TECHNICAL ANALYSIS":

    if st.session_state.df is None:
        st.info("📊 Load a stock from the sidebar to see the chart.")
    else:
        df = st.session_state.df
        result = st.session_state.signal_result

        # Company info header
        try:
            info = get_company_info(result.symbol)
            last_price = df["Close"].iloc[-1]
            prev_price = df["Close"].iloc[-2] if len(df) > 1 else last_price
            chg = pct_change(float(prev_price), float(last_price))

            ci1, ci2, ci3, ci4 = st.columns(4)
            ci1.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Company</div>
                    <div style="font-size:14px;font-weight:600;color:#e6edf3">{info.get('name', result.symbol)}</div>
                    <div style="font-size:11px;color:#8b949e;margin-top:4px;">{info.get('sector', '')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            ci2.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Last Price</div>
                    <div class="metric-value">₹{last_price:,.2f}</div>
                    <div class="metric-delta" style="color:{'#00e676' if chg >= 0 else '#f44336'}">
                        {'▲' if chg >= 0 else '▼'} {abs(chg):.2f}%
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            ci3.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">52-Week High</div>
                    <div class="metric-value">₹{info.get('52wHigh', 0):,.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            ci4.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">52-Week Low</div>
                    <div class="metric-value">₹{info.get('52wLow', 0):,.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        st.markdown("<br>", unsafe_allow_html=True)

        fig = build_price_chart(
            df,
            result.symbol,
            show_ema=show_ema,
            show_supertrend=show_supertrend,
            show_bollinger=show_bollinger,
            show_vwap=show_vwap,
            show_volume=show_volume,
            show_rsi=show_rsi,
            show_macd=show_macd,
            height=750,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Raw data expander
        with st.expander("📋 View Raw Data"):
            display_cols = [c for c in df.columns if c in [
                "Open", "High", "Low", "Close", "Volume",
                "EMA_20", "EMA_50", "RSI", "MACD", "ADX", "ATR",
                "Supertrend", "Supertrend_Direction",
            ]]
            raw_sub = df[display_cols].tail(50).copy()
            float_cols = raw_sub.select_dtypes(include=["float", "float64"]).columns
            st.dataframe(
                raw_sub.style.format({c: "{:.2f}" for c in float_cols}),
                use_container_width=True,
            )



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

elif active_tab == "QUANT BACKTEST":

    st.markdown("### ⚡ Strategy Backtesting")

    bt_col1, bt_col2 = st.columns([1, 2])

    with bt_col1:
        st.markdown("#### Configuration")
        bt_custom = st.text_input("Custom Ticker (Optional)", placeholder="e.g. WIPRO, SUZLON, TATAMOTORS", key="bt_custom_input")
        if bt_custom.strip():
            target_bt_symbol = normalise_symbol(bt_custom.strip())
        else:
            bt_symbol = st.selectbox(
                "Select Stock",
                options=[s["symbol"] for s in ALL_STOCKS],
                format_func=lambda s: f"{get_stock_name(s)} ({s})",
                key="bt_symbol",
            )
            target_bt_symbol = bt_symbol

        bt_period = st.selectbox("Period", ["1 Year", "2 Years", "3 Years", "5 Years"], key="bt_period")
        bt_period_map = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
        bt_capital = st.number_input(
            "Initial Capital (₹)", min_value=10_000, max_value=10_000_000,
            value=BT_DEFAULT_CAPITAL, step=10_000, format="%d", key="bt_capital",
        )
        run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    with bt_col2:
        if run_bt:
            with st.spinner(f"Running backtest for {target_bt_symbol}…"):
                try:
                    bt_df_raw = fetch_ohlcv(target_bt_symbol, interval="1d", period=bt_period_map[bt_period])
                    bt_df = compute_all_indicators(bt_df_raw)
                    bt_result = run_backtest(bt_df, initial_capital=bt_capital)


                    # Metrics row
                    bm1, bm2, bm3 = st.columns(3)
                    bm4, bm5, bm6 = st.columns(3)

                    def _bt_metric(col, label, val, color=None):
                        style = f"color:{color};" if color else ""
                        col.markdown(
                            f"""<div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value" style="font-size:20px;{style}">{val}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    color_np = "#00e676" if bt_result.net_profit >= 0 else "#f44336"
                    _bt_metric(bm1, "Net Profit", format_inr(bt_result.net_profit), color_np)
                    _bt_metric(bm2, "CAGR", f"{bt_result.cagr:.1f}%", color_np)
                    _bt_metric(bm3, "Sharpe Ratio", f"{bt_result.sharpe_ratio:.2f}")
                    _bt_metric(bm4, "Max Drawdown", f"{bt_result.max_drawdown:.1f}%", "#f44336")
                    _bt_metric(bm5, "Win Rate", f"{bt_result.win_rate:.1f}%")
                    _bt_metric(bm6, "Total Trades", str(bt_result.total_trades))

                    st.markdown("<br>", unsafe_allow_html=True)

                    m7, m8, m9 = st.columns(3)
                    _bt_metric(m7, "Profit Factor", f"{bt_result.profit_factor:.2f}")
                    _bt_metric(m8, "Sortino Ratio", f"{bt_result.sortino_ratio:.2f}")
                    _bt_metric(m9, "Avg Trade", format_inr(bt_result.avg_trade))

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Equity curve
                    if not bt_result.equity_curve.empty:
                        st.markdown("#### 📈 Equity Curve")
                        eq_fig = build_equity_curve(bt_result.equity_curve, bt_capital)
                        st.plotly_chart(eq_fig, use_container_width=True)

                    # Trade log
                    if not bt_result.trade_log.empty:
                        with st.expander("📋 Trade Log"):
                            st.dataframe(bt_result.trade_log, use_container_width=True)

                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
        else:
            st.markdown(
                """
                <div style="text-align:center;padding:60px 20px;color:#8b949e;">
                    <div style="font-size:48px;margin-bottom:16px;">⚡</div>
                    <p>Configure parameters and click <strong>Run Backtest</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

elif active_tab == "MARKET SCANNER":

    st.markdown("### 🌐 Overall Market Scanner & Sector Intelligence")

    sc_col1, sc_col2 = st.columns([1, 3])

    with sc_col1:
        scan_index = st.selectbox("Index / Sector Universe", list(INDEX_GROUPS.keys()), index=0, key="scan_index_select")
        scan_filter = st.multiselect(
            "Filter by Signal",
            ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
            default=["Strong Buy", "Buy"],
            key="scan_filter_select",
        )
        scan_min_confidence = st.slider("Min Confidence %", 0, 100, 35, key="scan_conf_slider")
        scan_display_limit = st.slider("Display Limit (Top Stocks)", 10, 100, 20, step=10, key="scan_limit_slider")
        run_scan = st.button("🔍 Scan Overall Market", type="primary", use_container_width=True, key="run_scan_btn")

    with sc_col2:
        if run_scan:
            scan_stocks = INDEX_GROUPS.get(scan_index, ALL_STOCKS)
            results_list = []

            progress = st.progress(0, text="Scanning Overall Market Data…")
            for i, stock in enumerate(scan_stocks):
                sym = stock["symbol"]
                progress.progress((i + 1) / len(scan_stocks), text=f"Scanning {stock['name']} ({sym})…")
                try:
                    raw = fetch_ohlcv(sym, interval="1d", period="6mo")
                    enriched = compute_all_indicators(raw)
                    sig_r = generate_signal(sym, enriched)
                    last = enriched.iloc[-1]
                    prev = enriched.iloc[-2] if len(enriched) > 1 else last
                    chg = pct_change(float(prev["Close"]), float(last["Close"]))
                    p1w = getattr(sig_r, "pct_1w", 0.0)
                    p2w = getattr(sig_r, "pct_2w", 0.0)

                    results_list.append({
                        "Symbol": sym,
                        "Name": stock["name"],
                        "Price": f"₹{last['Close']:,.2f}",
                        "Change%": f"{chg:+.2f}%",
                        "1W_Chg": f"{p1w:+.2f}%",
                        "2W_Chg": f"{p2w:+.2f}%",
                        "Signal": sig_r.signal,
                        "Confidence": sig_r.confidence,
                        "RSI": float(last.get("RSI", 0)) if pd.notna(last.get("RSI", None)) else 0.0,
                        "ADX": float(last.get("ADX", 0)) if pd.notna(last.get("ADX", None)) else 0.0,
                        "raw_change": chg,
                        "raw_1w": p1w,
                        "raw_2w": p2w,
                    })

                    # Persist scanned signal to SQLite database
                    save_signal(
                        symbol=sym,
                        signal=sig_r.signal,
                        confidence=sig_r.confidence,
                        entry_price=risk_r.entry_price,
                        stop_loss=risk_r.stop_loss,
                        take_profit=risk_r.take_profit,
                        risk_reward=risk_r.risk_reward,
                        interval="1d",
                        reasons=sig_r.reasons[:3],
                        source="Market Scanner",
                    )
                    log_search_event(
                        symbol=sym,
                        name=name_str,
                        signal=sig_r.signal,
                        confidence=sig_r.confidence,
                        price=risk_r.entry_price,
                        source="Market Scanner",
                    )

                except Exception as exc:

                    logger.warning("Scanner error for %s: %s", sym, exc)

            progress.empty()
            st.session_state.scan_results = results_list
            st.session_state.scan_index_label = scan_index

        if "scan_results" in st.session_state and st.session_state.scan_results:
            results_list = st.session_state.scan_results
            scan_index_label = st.session_state.get("scan_index_label", scan_index)

            # Calculate Overall Market Statistics
            total_scanned = len(results_list)
            strong_buy_cnt = sum(1 for r in results_list if r["Signal"] == "Strong Buy")
            buy_cnt = sum(1 for r in results_list if r["Signal"] == "Buy")
            hold_cnt = sum(1 for r in results_list if r["Signal"] == "Hold")
            sell_cnt = sum(1 for r in results_list if r["Signal"] in ("Sell", "Strong Sell"))
            bullish_pct = ((strong_buy_cnt + buy_cnt) / total_scanned * 100) if total_scanned > 0 else 0

            # Render Overall Market Breadth Banner
            st.markdown(
                f"""
                <div style="background:rgba(22,27,34,0.8); border:1px solid rgba(88,166,255,0.25); border-radius:12px; padding:14px 20px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div>
                            <span style="color:#8b949e; font-size:12px; text-transform:uppercase;">Overall Market Breadth:</span>
                            <span style="color:#00e676; font-weight:800; font-size:16px; margin-left:6px;">{bullish_pct:.1f}% Bullish Momentum</span>
                        </div>
                        <div style="font-size:12.5px; color:#c9d1d9;">
                            Scanned: <strong style="color:#f0f6fc;">{total_scanned} Stocks</strong> | 
                            🟢 Strong Buy: <strong style="color:#00e676;">{strong_buy_cnt}</strong> | 
                            🟩 Buy: <strong style="color:#58a6ff;">{buy_cnt}</strong> | 
                            ⚪ Hold: <strong style="color:#ffb300;">{hold_cnt}</strong> | 
                            🔴 Bearish: <strong style="color:#ff1744;">{sell_cnt}</strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Filter data based on user controls
            filtered_list = [
                r for r in results_list
                if (not scan_filter or r["Signal"] in scan_filter) and r["Confidence"] >= scan_min_confidence
            ]

            if len(filtered_list) >= scan_display_limit:
                display_items = sorted(filtered_list, key=lambda x: x["Confidence"], reverse=True)[:scan_display_limit]
            elif filtered_list:
                display_items = sorted(filtered_list, key=lambda x: x["Confidence"], reverse=True)
            else:
                display_items = sorted(results_list, key=lambda x: x["Confidence"], reverse=True)[:scan_display_limit]

            st.markdown(f"#### 🎯 Scanned Market Opportunities (Showing Top {len(display_items)} of {total_scanned} Scanned)")
            for idx, item in enumerate(display_items):
                sig_c = SIGNAL_COLORS.get(item["Signal"], "#9e9e9e")
                chg_c = "#00e676" if item["raw_change"] >= 0 else "#ff1744"
                chg_1w_c = "#00e676" if item.get("raw_1w", 0) >= 0 else "#ff1744"
                chg_2w_c = "#00e676" if item.get("raw_2w", 0) >= 0 else "#ff1744"

                col_info, col_act = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left:4px solid {sig_c}; padding:10px 16px; margin-bottom:4px; display:flex; justify-content:space-between; align-items:center;">
                            <div style="display:flex; align-items:center; gap:12px;">
                                <span class="mono-font" style="font-weight:700; color:#f0f6fc; font-size:14px;">{item['Symbol']}</span>
                                <span style="font-size:12px; color:#8b949e;">{item['Name']}</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:14px;">
                                <span class="mono-font" style="font-weight:700; color:#f0f6fc; font-size:13px;">{item['Price']}</span>
                                <span class="mono-font" style="font-size:12px; color:{chg_c}; font-weight:600;">1D: {item['Change%']}</span>
                                <span class="mono-font" style="font-size:12px; color:{chg_1w_c}; font-weight:600;">1W: {item.get('1W_Chg', '0.00%')}</span>
                                <span class="mono-font" style="font-size:12px; color:{chg_2w_c}; font-weight:600;">2W: {item.get('2W_Chg', '0.00%')}</span>
                                <span style="font-size:11px; font-weight:700; color:{sig_c}; background:{sig_c}22; padding:2px 8px; border-radius:8px;">{item['Signal']}</span>
                                <span class="mono-font" style="font-size:12px; color:#58a6ff; font-weight:700;">{item['Confidence']:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                    with col_act:
                        if st.button(f"⚡ Analyse", key=f"scan_btn_{idx}_{item['Symbol']}"):
                            load_and_analyse(item['Symbol'])
                            st.rerun()
        else:
            st.markdown(
                """
                <div style="text-align:center;padding:60px 20px;color:#8b949e;">
                    <div style="font-size:48px;margin-bottom:16px;">🌐</div>
                    <h3 style="color:#58a6ff; margin-bottom:8px;">Overall Market Scanner</h3>
                    <p>Select <strong>ALL STOCKS (100+ Liquid NSE)</strong> or a specific sector and click <strong>Scan Overall Market</strong> to analyze real-time market breadth and find high-confidence signals.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SECTOR & INDUSTRY PERFORMANCE (1-MONTH TREND FORECASTING)
# ═══════════════════════════════════════════════════════════════════════════════

elif active_tab == "SECTOR & INDUSTRY PERFORMANCE":
    st.markdown("### 🏢 Industry Sector Performance & 1-Month Trend Forecasting")
    st.markdown(
        """
        <div class="ai-box" style="padding:14px 20px; margin-bottom:18px; border-radius:12px;">
            <div style="font-size:13.5px; line-height:1.5;">
                Track real-time momentum, 1-month predictive trend forecasts, and sector rotation across key Indian market industries.
                Quantitative scoring incorporates <strong>30-Day Returns, Moving Average Breadth (% above 20 EMA), RSI Confluence, and Outperforming Equities</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn, col_space = st.columns([1.5, 3])
    with col_btn:
        run_sector_scan = st.button("⚡ Scan Industry Sectors & 1M Forecast", type="primary", use_container_width=True)

    if run_sector_scan or "sector_analysis_data" not in st.session_state:
        with st.spinner("Analyzing Indian Industry Sectors & Computing 30-Day Trend Models..."):
            sector_data = analyze_sector_performance()
            st.session_state["sector_analysis_data"] = sector_data

    sector_data = st.session_state.get("sector_analysis_data", [])

    if sector_data:
        top_sector = sector_data[0]
        t_color = top_sector["trend_color"]
        t_sec = top_sector["sector"]
        t_ret = top_sector["ret_1m"]
        st.markdown(f"#### 🏆 Top Outperforming Industry: <span style='color:{t_color};'>{t_sec} ({t_ret:+.2f}% 1M)</span>", unsafe_allow_html=True)


        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f"""<div class="metric-card" style="border-top:3px solid {top_sector['trend_color']};">
                <div class="metric-label">Leading Sector</div>
                <div style="font-size:18px; font-weight:800; color:#f0f6fc;">{top_sector['sector']}</div>
                <div style="font-size:12px; color:{top_sector['trend_color']}; margin-top:2px;">{top_sector['trend_icon']} {top_sector['trend_label']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">1-Month Return</div>
                <div style="font-size:22px; font-weight:800; color:{'#00e676' if top_sector['ret_1m']>=0 else '#ff1744'};">{top_sector['ret_1m']:+.2f}%</div>
                <div style="font-size:11px; color:#8b949e; margin-top:2px;">1W: {top_sector['ret_1w']:+.2f}% | 1D: {top_sector['ret_1d']:+.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">1M Forecast Target</div>
                <div style="font-size:20px; font-weight:800; color:#58a6ff;">{top_sector['target_range']}</div>
                <div style="font-size:11px; color:#8b949e; margin-top:2px;">Score: {top_sector['score_1m']:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        c4.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Sector Breadth</div>
                <div style="font-size:20px; font-weight:800; color:#00e676;">{top_sector['pct_above_ema20']:.0f}% > 20 EMA</div>
                <div style="font-size:11px; color:#8b949e; margin-top:2px;">RSI: {top_sector['avg_rsi']:.1f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Complete Industry Performance & 1-Month Trend Matrix")

        matrix_rows = []
        for s in sector_data:
            leaders_str = ", ".join([f"{l['symbol'].replace('.NS','')} ({l['ret_1m']:+.1f}%)" for l in s['top_leaders']])
            matrix_rows.append({
                "Industry Sector": s["sector"],
                "1M Return": f"{s['ret_1m']:+.2f}%",
                "1W Return": f"{s['ret_1w']:+.2f}%",
                "1D Return": f"{s['ret_1d']:+.2f}%",
                "1-Month Trend Forecast": f"{s['trend_icon']} {s['trend_label']}",
                "30-Day Target Range": s["target_range"],
                "Sector Breadth (>20 EMA)": f"{s['pct_above_ema20']:.0f}%",
                "Top Outperforming Leaders": leaders_str,
            })

        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🏢 Detailed Sector Breakdown & 30-Day Outlook Cards")

        for sec in sector_data:
            with st.expander(f"{sec['trend_icon']} {sec['sector']} — 1M Return: {sec['ret_1m']:+.2f}% | Forecast: {sec['trend_label']}"):
                st.markdown(f"**30-Day Outlook Note:** {sec['outlook_text']}")
                st.markdown(f"**Expected 30-Day Price Movement Target:** <span style='color:{sec['trend_color']}; font-weight:800;'>{sec['target_range']}</span>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**⭐ Top Performing Equities in this Sector:**")

                l_cols = st.columns(len(sec["top_leaders"]))
                for idx, leader in enumerate(sec["top_leaders"]):
                    with l_cols[idx]:
                        st.markdown(
                            f"""
                            <div class="metric-card" style="padding:12px; border:1px solid rgba(88,166,255,0.25);">
                                <div style="font-weight:800; font-size:14px; color:#58a6ff;">{leader['symbol'].replace('.NS','')}</div>
                                <div style="font-size:11px; color:#8b949e; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{leader['name']}</div>
                                <div style="font-size:14px; font-weight:800; color:{'#00e676' if leader['ret_1m']>=0 else '#ff1744'}; margin-top:4px;">1M: {leader['ret_1m']:+.2f}%</div>
                                <div style="font-size:11px; color:#c9d1d9;">Price: ₹{leader['price']:,.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Analyse {leader['symbol'].replace('.NS','')}", key=f"sec_lead_{sec['sector']}_{idx}"):
                            load_and_analyse(leader['symbol'])
                            st.rerun()



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════

elif active_tab == "WATCHLIST & AUDIT LOGS":


    st.markdown("### ⭐ Watchlist")

    wl_add_col, wl_table_col = st.columns([1, 3])

    with wl_add_col:
        st.markdown("#### Add Stock")
        wl_new = st.text_input("Symbol (e.g. WIPRO)", key="wl_new_symbol")
        wl_name = st.text_input("Name (optional)", key="wl_new_name")
        wl_exch = st.selectbox("Exchange", ["NSE", "BSE"], key="wl_exch")
        if st.button("➕ Add to Watchlist", use_container_width=True):
            if wl_new.strip():
                sym = normalise_symbol(wl_new.strip(), wl_exch)
                add_to_watchlist(sym, wl_name or sym, wl_exch)
                st.success(f"Added {sym}")
                st.rerun()
            else:
                st.warning("Enter a symbol first.")

    with wl_table_col:
        watchlist = get_watchlist()
        if not watchlist:
            st.markdown(
                """
                <div style="text-align:center;padding:60px;color:#8b949e;">
                    <div style="font-size:48px;margin-bottom:16px;">⭐</div>
                    <p>Your watchlist is empty. Add stocks using the panel on the left.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Live signals for watchlist
            wl_refresh = st.button("Refresh Signals", key="wl_refresh")
            if wl_refresh:
                with st.spinner("Fetching latest signals…"):
                    wl_data = []
                    for item in watchlist:
                        sym = item["symbol"]
                        try:
                            wl_df = fetch_ohlcv(sym, interval="1d", period="6mo")
                            wl_df = compute_all_indicators(wl_df)
                            sig_r = generate_signal(sym, wl_df)
                            last = wl_df.iloc[-1]
                            prev = wl_df.iloc[-2]
                            chg = pct_change(float(prev["Close"]), float(last["Close"]))
                            wl_data.append({
                                "Symbol": sym,
                                "Name": item["name"],
                                "Price": f"₹{last['Close']:,.2f}",
                                "Change %": f"{chg:+.2f}%",
                                "Signal": f"{SIGNAL_EMOJI.get(sig_r.signal, '')} {sig_r.signal}",
                                "Confidence": f"{sig_r.confidence:.1f}%",
                                "Added": item["added_at"][:10],
                            })
                        except Exception:
                            wl_data.append({"Symbol": sym, "Name": item["name"], "Signal": "Error"})

                    st.dataframe(pd.DataFrame(wl_data), use_container_width=True, hide_index=True)
            else:
                # Show basic list with 1-click Analyse & Remove buttons
                for item in watchlist:

                    col_a, col_b, col_c, col_d = st.columns([2, 2.5, 1.5, 1])
                    col_a.markdown(f"**{item['symbol']}**")
                    col_b.markdown(f"<span style='color:#8b949e'>{item['name'] or '—'}</span>", unsafe_allow_html=True)
                    if col_c.button("Analyse", key=f"wl_ana_{item['symbol']}"):
                        load_and_analyse(item["symbol"])
                        st.rerun()
                    if col_d.button("Remove", key=f"rm_{item['symbol']}"):
                        remove_from_watchlist(item["symbol"])
                        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📜 Full Signal Audit Log & Quantitative Database Records")
    recent = get_recent_signals(limit=50)
    if recent:
        sig_df = pd.DataFrame(recent)
        
        # Download full audit log CSV button
        csv_data = sig_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Signal Audit Log (CSV)",
            data=csv_data,
            file_name=f"signals_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_signals_csv",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for idx, r in enumerate(recent[:20]):
            sym = r["symbol"]
            sig = r["signal"]
            conf = r.get("confidence", 0.0)
            entry = r.get("entry_price", 0.0)
            sl = r.get("stop_loss", 0.0)
            tp = r.get("take_profit", 0.0)
            rr = r.get("risk_reward", 0.0)
            src = r.get("source", "Signal Terminal")
            dt_str = str(r.get("generated_at", ""))[:16]
            sig_c = SIGNAL_COLORS.get(sig, "#9e9e9e")

            with st.expander(f"{sym} — {sig} ({conf:.1f}% Conf) | Entry: ₹{entry:,.2f} | {dt_str}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown(f"**Signal:** <span style='color:{sig_c}; font-weight:700;'>{sig}</span>", unsafe_allow_html=True)
                col2.markdown(f"**Entry Price:** ₹{entry:,.2f}")
                col3.markdown(f"**Stop Loss:** ₹{sl:,.2f}")
                col4.markdown(f"**Take Profit:** ₹{tp:,.2f}")

                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"**Confidence:** {conf:.1f}%")
                m2.markdown(f"**Risk:Reward:** 1:{rr:.1f}")
                m3.markdown(f"**Source:** `{src}`")
                m4.markdown(f"**Generated:** `{dt_str}`")

                if r.get("reasons"):
                    st.markdown("**Signal Reasons:**")
                    try:
                        import json
                        reasons_list = json.loads(r["reasons"]) if isinstance(r["reasons"], str) and r["reasons"].startswith("[") else [r["reasons"]]
                        for re in reasons_list:
                            st.markdown(f"- {re}")
                    except Exception:
                        st.markdown(f"- {r['reasons']}")

                if st.button(f"Analyse {sym}", key=f"hist_btn_{idx}_{sym}"):
                    load_and_analyse(sym)
                    st.rerun()


    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("### Search History & End-Of-Day (EOD) Analytics")

    eod_col1, eod_col2 = st.columns([1, 2.5])

    with eod_col1:
        eod_date = st.date_input("Select Date", datetime.now(), key="eod_date_picker")
        eod_date_str = eod_date.strftime("%Y-%m-%d")
        eod_stats = get_eod_summary(eod_date_str)

        st.markdown(
            f"""
            <div class="metric-card" style="border-top:3px solid #58a6ff; margin-bottom:12px;">
                <div style="font-size:12px; color:#8b949e; text-transform:uppercase;">Total Queries ({eod_date_str})</div>
                <div style="font-size:28px; font-weight:800; color:#58a6ff; margin-top:2px;">{eod_stats['total_queries']}</div>
                <div style="font-size:12px; color:#8b949e; margin-top:4px;">Unique Stocks: <strong style="color:#f0f6fc;">{eod_stats['unique_stocks']}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if eod_stats["top_searched"]:
            st.markdown("<strong style='font-size:13px; color:#c9d1d9;'>Most Searched Today</strong>", unsafe_allow_html=True)

            for s_item in eod_stats["top_searched"]:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; font-size:12.5px; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                        <span style="color:#f0f6fc; font-weight:600;">{s_item['symbol']}</span>
                        <span style="color:#58a6ff; font-weight:700;">{s_item['query_count']} queries</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with eod_col2:
        search_logs = get_search_history(limit=50, date_filter=eod_date_str)
        if search_logs:
            sh_df = pd.DataFrame(search_logs)

            # Format download CSV
            csv_cols = [c for c in ["searched_at", "symbol", "name", "signal", "confidence", "price", "source"] if c in sh_df.columns]
            csv_data = sh_df[csv_cols].to_csv(index=False)
            st.download_button(
                label=f"📥 Download {eod_date_str} EOD Search Log (CSV)",
                data=csv_data,
                file_name=f"StockSense_EOD_Search_Log_{eod_date_str}.csv",
                mime="text/csv",
                key="download_eod_csv",
            )

            for s_idx, row in enumerate(search_logs[:20]):
                s_sym = row["symbol"]
                s_sig = row.get("signal", "N/A")
                s_conf = row.get("confidence", 0.0)
                s_time = str(row.get("searched_at", ""))[:16]
                s_c = SIGNAL_COLORS.get(s_sig, "#9e9e9e")

                sc1, sc2, sc3, sc4, sc5 = st.columns([2, 2, 1.5, 2, 1.5])
                sc1.markdown(f"**{s_sym}**")
                sc2.markdown(f"<span style='color:{s_c}; font-weight:700;'>{s_sig}</span>", unsafe_allow_html=True)
                sc3.markdown(f"<span class='mono-font'>{s_conf:.1f}%</span>", unsafe_allow_html=True)
                sc4.markdown(f"<span style='color:#8b949e; font-size:12px;'>{s_time}</span>", unsafe_allow_html=True)
                if sc5.button("⚡ Re-Analyse", key=f"sh_btn_{s_idx}_{s_sym}"):
                    load_and_analyse(s_sym)
                    st.rerun()
        else:
            st.info(f"No search events logged for {eod_date_str} yet. Searches made today will automatically appear here.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("### 👥 Registered User Profiles & Access Log")
    user_list = get_all_users()
    if user_list:
        with st.expander(f"📋 View Registered App Users ({len(user_list)} User Record{'s' if len(user_list) > 1 else ''})"):
            u_df = pd.DataFrame(user_list)
            st.dataframe(u_df[["id", "name", "email", "phone", "created_at", "last_login"]], use_container_width=True)





# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — INSTITUTIONAL EQUITY RESEARCH (LLM & GEMINI INTELLIGENCE)
# ═══════════════════════════════════════════════════════════════════════════════

elif active_tab == "INSTITUTIONAL RESEARCH":

    st.markdown("### 🏛️ Institutional Equity Research Engine (Gemini / LLM Intelligence)")
    st.markdown(
        """
        <div class="ai-box" style="padding:14px 20px; margin-bottom:16px; border-radius:12px;">
            <div style="font-size:13.5px; line-height:1.5;">
                Act like a disciplined Indian equity research analyst using <strong>verifiable public information</strong> (annual reports, concalls, BSE/NSE filings).
                Generate institutional-grade research across <strong>13 distinct prompts</strong> (Business Model, Moat Score, DCF Sandbox, Downside Risk Ranking, FII/DII Thesis, Bull/Bear Debate, Governance Scorecard, and Peer Tables).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    r_col1, r_col2 = st.columns([1, 2.5])

    with r_col1:
        st.markdown("#### ⚙️ Research Controls")

        res_default_sym = st.session_state.get("selected_symbol", "RELIANCE.NS")
        res_default_name = st.session_state.get("company_name", get_stock_name(res_default_sym))

        research_company_input = st.text_input(
            "Company Name or Ticker Symbol",
            value=res_default_name if res_default_name else res_default_sym,
            placeholder="e.g. Wipro, Reliance, Tata Motors, TCS, Infosys, HAL",
            help="Type any Indian company name or NSE ticker symbol to generate institutional research.",
            key="research_company_name_input",
        )

        # Resolve symbol & company name dynamically from user input
        if research_company_input.strip():
            raw_input = research_company_input.strip()
            res_symbol = normalise_symbol(raw_input, "NSE")
            res_comp_name = get_stock_name(res_symbol)
            if res_comp_name == res_symbol and len(raw_input) > 2:
                res_comp_name = raw_input.title()
        else:
            res_symbol = res_default_sym
            res_comp_name = res_default_name

        st.markdown(
            f"""
            <div style="background:rgba(88,166,255,0.08); border:1px solid rgba(88,166,255,0.2); border-radius:8px; padding:6px 12px; margin-bottom:12px; font-size:12.5px;">
                Target: <strong style="color:#58a6ff;">{res_comp_name}</strong> <span style="color:#8b949e;">({res_symbol})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        llm_provider = st.selectbox(
            "AI Intelligence Engine",
            ["Google Gemini 1.5 Flash (Recommended)", "OpenAI GPT-4o / GPT-4o-mini", "Built-in Financial Intelligence Engine (No API Key Required)"],
            index=0,
            key="llm_provider_select",
        )

        llm_api_key = st.text_input(
            "API Key (Optional)",
            type="password",
            value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("OPENAI_API_KEY", ""))),
            help="Enter your Gemini or OpenAI API Key. If left blank, the Built-in Engine will be used.",
            key="llm_api_key_input",
        )

        report_mode = st.radio(
            "Report Generation Mode",
            ["🚀 Full 13-Module Institutional Report", "🎯 Single Prompt Analysis"],
            index=0,
            key="report_mode_radio",
        )

        selected_prompt_key = "P1"
        if report_mode == "🎯 Single Prompt Analysis":
            prompt_choices = {k: v["title"] for k, v in INSTITUTIONAL_PROMPTS.items()}
            selected_prompt_key = st.selectbox(
                "Select Prompt Module",
                options=list(prompt_choices.keys()),
                format_func=lambda k: prompt_choices[k],
                key="single_prompt_select",
            )
            st.caption(INSTITUTIONAL_PROMPTS[selected_prompt_key]["description"])

        gen_report_btn = st.button("⚡ Generate Institutional Research Report", type="primary", use_container_width=True, key="gen_report_btn")

    with r_col2:
        if gen_report_btn:
            with st.spinner(f"Generating Institutional Research for {res_comp_name} ({res_symbol})…"):
                # Fetch fresh live OHLCV & indicators dynamically for the target stock
                try:
                    df_res = fetch_ohlcv(res_symbol, interval="1d", period="6mo")
                    df_res = compute_all_indicators(df_res)
                    sig_res = generate_signal(res_symbol, df_res)
                    risk_res = calculate_risk(df_res, sig_res.signal)
                    curr_price = float(df_res["Close"].iloc[-1])
                    curr_sig = sig_res.signal
                    curr_conf = float(sig_res.confidence)
                    curr_rsi = float(df_res["RSI"].iloc[-1]) if "RSI" in df_res.columns else 50.0
                    curr_adx = float(df_res["ADX"].iloc[-1]) if "ADX" in df_res.columns else 20.0
                except Exception as exc:
                    logger.warning("Error computing indicators for %s: %s", res_symbol, exc)
                    try:
                        df_simple = fetch_ohlcv(res_symbol, interval="1d", period="1mo")
                        curr_price = float(df_simple["Close"].iloc[-1])
                    except Exception:
                        curr_price = 0.0
                    curr_sig = "Hold"
                    curr_conf = 60.0
                    curr_rsi = 50.0
                    curr_adx = 20.0


                comp_info = get_company_info(res_symbol)


                metrics_pack = {
                    "price": curr_price,
                    "signal": curr_sig,
                    "confidence": curr_conf,
                    "rsi": curr_rsi,
                    "adx": curr_adx,
                    "sector": comp_info.get("sector", "Equities"),
                    "pe": str(comp_info.get("pe", "24.5")),
                    "marketCap": comp_info.get("marketCap", 0),
                }

                generated_outputs = {}
                keys_to_run = list(INSTITUTIONAL_PROMPTS.keys()) if report_mode == "🚀 Full 13-Module Institutional Report" else [selected_prompt_key]

                for p_key in keys_to_run:
                    p_info = INSTITUTIONAL_PROMPTS[p_key]
                    prompt_formatted = p_info["template"].format(company_name=res_comp_name, symbol=res_symbol)

                    res_text = ""
                    if "Gemini" in llm_provider and llm_api_key.strip():
                        try:
                            res_text = call_gemini_api(prompt_formatted, llm_api_key.strip())
                        except Exception as exc:
                            st.warning(f"Gemini API fallback for {p_key}: {exc}")
                            res_text = generate_fallback_institutional_report(p_key, res_symbol, res_comp_name, metrics_pack)
                    elif "OpenAI" in llm_provider and llm_api_key.strip():
                        try:
                            res_text = call_openai_api(prompt_formatted, llm_api_key.strip())
                        except Exception as exc:
                            st.warning(f"OpenAI API fallback for {p_key}: {exc}")
                            res_text = generate_fallback_institutional_report(p_key, res_symbol, res_comp_name, metrics_pack)
                    else:
                        res_text = generate_fallback_institutional_report(p_key, res_symbol, res_comp_name, metrics_pack)

                    generated_outputs[p_key] = res_text

                st.session_state.institutional_report_data = {
                    "symbol": res_symbol,
                    "company_name": res_comp_name,
                    "outputs": generated_outputs,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

        if "institutional_report_data" in st.session_state and st.session_state.institutional_report_data:
            rep = st.session_state.institutional_report_data
            outputs = rep["outputs"]

            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div>
                        <span style="font-size:18px; font-weight:800; color:#58a6ff;">🏛️ Research Report: {rep['company_name']} ({rep['symbol']})</span>
                        <div style="font-size:11px; color:#8b949e;">Generated: {rep['generated_at']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            full_md_text = f"# 🏛️ Institutional Equity Research Report: {rep['company_name']} ({rep['symbol']})\n"
            full_md_text += f"**Date**: {rep['generated_at']} | **Analyst Lens**: Verifiable Public Filings & Institutional Mandate\n\n"
            for k, text in outputs.items():
                title = INSTITUTIONAL_PROMPTS[k]["title"]
                full_md_text += f"## {title}\n{text}\n\n---\n\n"

            st.download_button(
                label="📥 Download Complete Equity Research Report (.md)",
                data=full_md_text,
                file_name=f"Institutional_Research_{rep['symbol']}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                key="download_research_md",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            for k, text in outputs.items():
                title = INSTITUTIONAL_PROMPTS[k]["title"]
                with st.expander(f"📌 {title}", expanded=True):
                    st.markdown(text, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div style="text-align:center;padding:70px 20px;color:#8b949e;">
                    <div style="font-size:52px;margin-bottom:16px;">🏛️</div>
                    <h3 style="color:#58a6ff; margin-bottom:8px;">Institutional Equity Research Sandbox</h3>
                    <p>Select your research parameters and click <strong>Generate Institutional Research Report</strong> to run all 13 prompts (Business Model, Moat Scorecard, Valuation DCF, Risk Ranking, Bull/Bear Debate, Governance Scorecard, and Peer Comparison).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

elif active_tab == "ALERT MANAGER":


    st.markdown("### 🔔 Alerts & Notifications")

    al1, al2 = st.columns(2)

    with al1:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#161b22,#1c2333);
                        border:1px solid rgba(88,166,255,0.2);border-radius:12px;padding:20px;">
                <h4 style="color:#29b6f6;margin:0 0 12px;">📱 Telegram Setup</h4>
                <ol style="color:#8b949e;font-size:13px;padding-left:16px;">
                    <li>Create a Telegram bot via <strong>@BotFather</strong></li>
                    <li>Copy the bot token</li>
                    <li>Start a chat with your bot and get the chat ID</li>
                    <li>Enter credentials in the sidebar → Alert Settings</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with al2:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#161b22,#1c2333);
                        border:1px solid rgba(88,166,255,0.2);border-radius:12px;padding:20px;">
                <h4 style="color:#ff9800;margin:0 0 12px;">📧 Email Setup</h4>
                <ol style="color:#8b949e;font-size:13px;padding-left:16px;">
                    <li>Use a Gmail account</li>
                    <li>Enable <strong>App Passwords</strong> in Google Account</li>
                    <li>Generate an app-specific password</li>
                    <li>Enter credentials in the sidebar → Alert Settings</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🧪 Test Alerts")
    if st.session_state.signal_result:
        result = st.session_state.signal_result
        risk = st.session_state.risk
        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button("Test Telegram", use_container_width=True):
                msg = format_signal_message(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                )
                ok = send_telegram_alert(msg, chat_id=tg_chat or None)
                st.success("✅ Sent!") if ok else st.error("❌ Failed (check credentials).")
        with tc2:
            if st.button("Test Email", use_container_width=True):
                subj, body = format_signal_email(
                    result.symbol, result.signal, result.confidence,
                    risk.entry_price, risk.stop_loss, risk.take_profit, risk.risk_reward,
                    result.reasons,
                )
                ok = send_email_alert(subj, body, receiver=email_to or None)
                st.success("✅ Sent!") if ok else st.error("❌ Failed (check credentials).")
    else:
        st.info("Load a stock first to test alerts.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:rgba(22,27,34,0.8);border-radius:12px;padding:20px;">
            <h4 style="color:#8b949e;margin:0 0 8px;">🔮 Future Alert Triggers</h4>
            <ul style="color:#8b949e;font-size:13px;">
                <li>New Buy / Sell signal detected</li>
                <li>Target price hit</li>
                <li>Stop loss triggered</li>
                <li>RSI overbought / oversold</li>
                <li>Volume spike detected</li>
                <li>Supertrend flip</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div style="text-align:center;padding:40px 20px 20px;color:#484f58;font-size:12px;">
        <div style="margin-bottom:8px;">
            {APP_NAME} v{APP_VERSION} — Built with ❤️ using Streamlit & Python
        </div>
        <div>
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only.
            Not financial advice. Always do your own research.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Auto-refresh rerun
if auto_refresh:
    time.sleep(1)
    st.rerun()


