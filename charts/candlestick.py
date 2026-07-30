"""
charts/candlestick.py
=====================
Plotly chart builder for price, indicators, and volume.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_price_chart(
    df: pd.DataFrame,
    symbol: str,
    show_ema: bool = True,
    show_supertrend: bool = True,
    show_bollinger: bool = True,
    show_vwap: bool = True,
    show_volume: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    height: int = 700,
) -> go.Figure:
    """
    Build an interactive Plotly chart with candlestick, overlays, and subplots.

    Args:
        df:               Enriched OHLCV DataFrame.
        symbol:           Stock symbol for the title.
        show_ema:         Overlay EMA lines.
        show_supertrend:  Overlay Supertrend.
        show_bollinger:   Overlay Bollinger Bands.
        show_vwap:        Overlay VWAP.
        show_volume:      Add Volume subplot.
        show_rsi:         Add RSI subplot.
        show_macd:        Add MACD subplot.
        height:           Total chart height in pixels.

    Returns:
        Plotly Figure.
    """
    # ── Subplot layout ─────────────────────────────────────────────────────
    rows = 1
    row_heights = [0.55]
    subplot_titles = [symbol]

    if show_volume:
        rows += 1
        row_heights.append(0.12)
        subplot_titles.append("Volume")
    if show_rsi:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append("RSI (14)")
    if show_macd:
        rows += 1
        row_heights.append(0.18)
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Candlestick ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#00e676",
            decreasing_line_color="#f44336",
            increasing_fillcolor="#00e676",
            decreasing_fillcolor="#f44336",
            opacity=0.9,
        ),
        row=1,
        col=1,
    )

    # ── EMA overlays ────────────────────────────────────────────────────────
    if show_ema:
        ema_configs = [
            ("EMA_20", "#ffd740", "EMA 20", 1.5),
            ("EMA_50", "#29b6f6", "EMA 50", 1.5),
            ("EMA_200", "#ef5350", "EMA 200", 2.0),
        ]
        for col_name, color, label, width in ema_configs:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col_name],
                        name=label,
                        line=dict(color=color, width=width),
                        opacity=0.8,
                    ),
                    row=1,
                    col=1,
                )

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    if show_bollinger and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(150,150,255,0.05)",
            ),
            row=1,
            col=1,
        )

    # ── VWAP ───────────────────────────────────────────────────────────────
    if show_vwap and "VWAP" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["VWAP"],
                name="VWAP",
                line=dict(color="#ff9800", width=1.5, dash="dashdot"),
                opacity=0.85,
            ),
            row=1,
            col=1,
        )

    # ── Supertrend ──────────────────────────────────────────────────────────
    if show_supertrend and "Supertrend" in df.columns:
        st_buy = df[df["Supertrend_Direction"] == 1]
        st_sell = df[df["Supertrend_Direction"] == -1]

        fig.add_trace(
            go.Scatter(
                x=st_buy.index,
                y=st_buy["Supertrend"],
                name="Supertrend Buy",
                mode="markers",
                marker=dict(color="#00e676", size=3, symbol="circle"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=st_sell.index,
                y=st_sell["Supertrend"],
                name="Supertrend Sell",
                mode="markers",
                marker=dict(color="#f44336", size=3, symbol="circle"),
            ),
            row=1,
            col=1,
        )

    # ── Buy / Sell markers ───────────────────────────────────────────────────
    if "Signal_Label" in df.columns:
        buy_df = df[df["Signal_Label"].isin(["Strong Buy", "Buy"])]
        sell_df = df[df["Signal_Label"].isin(["Strong Sell", "Sell"])]

        if not buy_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_df.index,
                    y=buy_df["Low"] * 0.99,
                    name="Buy Signal",
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#00e676",
                        line=dict(color="#ffffff", width=1),
                    ),
                ),
                row=1,
                col=1,
            )
        if not sell_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_df.index,
                    y=sell_df["High"] * 1.01,
                    name="Sell Signal",
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#f44336",
                        line=dict(color="#ffffff", width=1),
                    ),
                ),
                row=1,
                col=1,
            )

    # ── Volume subplot ───────────────────────────────────────────────────────
    current_row = 2
    if show_volume:
        colors = [
            "#00e676" if c >= o else "#f44336"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        if "Volume_SMA" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Volume_SMA"],
                    name="Vol SMA",
                    line=dict(color="#ffd740", width=1),
                    showlegend=False,
                ),
                row=current_row,
                col=1,
            )
        current_row += 1

    # ── RSI subplot ──────────────────────────────────────────────────────────
    if show_rsi and "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI",
                line=dict(color="#ba68c8", width=1.5),
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        # Overbought / Oversold lines
        for level, color in [(70, "rgba(244,67,54,0.4)"), (30, "rgba(0,230,118,0.4)")]:
            fig.add_hline(
                y=level,
                line=dict(color=color, width=1, dash="dash"),
                row=current_row,
                col=1,
            )
        fig.add_hline(
            y=50,
            line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
            row=current_row,
            col=1,
        )
        current_row += 1

    # ── MACD subplot ─────────────────────────────────────────────────────────
    if show_macd and "MACD" in df.columns:
        hist_colors = [
            "#00e676" if v >= 0 else "#f44336"
            for v in df["MACD_Hist"]
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                name="MACD Hist",
                marker_color=hist_colors,
                opacity=0.6,
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD",
                line=dict(color="#29b6f6", width=1.5),
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                name="Signal",
                line=dict(color="#ff9800", width=1.5),
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )

    # ── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(9,12,16,0.95)",
        plot_bgcolor="rgba(13,17,23,0.95)",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12, color="#c9d1d9"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="rgba(88,166,255,0.2)",
            borderwidth=1,
            font=dict(size=11, color="#8b949e"),
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=15, r=15, t=40, b=15),
        hovermode="x unified",
    )

    # Axis styling
    for i in range(1, rows + 1):
        fig.update_xaxes(
            row=i,
            col=1,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        )
        fig.update_yaxes(
            row=i,
            col=1,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            tickformat=".2f" if i == 1 else None,
        )

    return fig


def build_equity_curve(equity_series: pd.Series, initial_capital: float) -> go.Figure:
    """
    Build an equity curve chart for backtesting results.

    Args:
        equity_series:   Series of equity values indexed by date.
        initial_capital: Starting capital.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Gradient fill
    fig.add_trace(
        go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            name="Portfolio Value",
            line=dict(color="#00e676", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.1)",
            hovertemplate="₹%{y:,.0f}<extra></extra>",
        )
    )

    # Initial capital reference line
    fig.add_hline(
        y=initial_capital,
        line=dict(color="rgba(255,215,64,0.5)", width=1, dash="dash"),
        annotation_text=f"Capital: ₹{initial_capital:,.0f}",
        annotation_position="left",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(13,17,23,1)",
        plot_bgcolor="rgba(13,17,23,1)",
        font=dict(family="Inter, sans-serif", size=12, color="#e0e0e0"),
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickformat="₹,.0f",
        ),
        hovermode="x unified",
    )
    return fig
