"""
Reusable dashboard components: price cards, risk charts, rule-based screen.
Real data only — no simulated "live" ticks.
"""

import html
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.market_utils import format_price
from src.app.utils import safe_float

COLORS = ["#6366f1", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#06b6d4", "#ef4444", "#84cc16"]


def render_price_cards(tickers: List[str], prices: Dict[str, float],
                       history: Optional[Dict[str, list]] = None,
                       sources: Optional[Dict[str, str]] = None, max_cards: int = 8):
    shown = tickers[:max_cards]
    if not shown:
        return
    cols = st.columns(len(shown))
    for col, t in zip(cols, shown):
        hist = (history or {}).get(t, [])
        change_html = '<span style="color:#64748b;">—</span>'
        if len(hist) >= 2 and hist[-2]["price"] > 0:
            ch = (hist[-1]["price"] / hist[-2]["price"] - 1) * 100
            color = "#10b981" if ch >= 0 else "#ef4444"
            change_html = (f'<span style="color:{color};">{"↑" if ch >= 0 else "↓"} {ch:+.2f}%</span>'
                           '<span style="color:#64748b;font-size:0.7rem;"> since last refresh</span>')
        src = html.escape((sources or {}).get(t, "") or "")
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:0.8rem;
                    text-align:center;border:1px solid rgba(255,255,255,0.1);">
            <div style="font-size:0.8rem;color:#94a3b8;">{html.escape(t)}</div>
            <div style="font-size:1.2rem;font-weight:600;">{html.escape(format_price(prices.get(t), t))}</div>
            <div style="font-size:0.85rem;">{change_html}</div>
            <div style="font-size:0.65rem;color:#64748b;">{src}</div>
        </div>""", unsafe_allow_html=True)


def render_price_chart(ticker: str, history: List[Dict], height: int = 400):
    if not history:
        st.info(f"No price history for {ticker} yet")
        return
    df = pd.DataFrame(history)
    fig = go.Figure(go.Scatter(x=df["time"], y=df["price"], mode="lines+markers", name=ticker,
                               line=dict(color="#6366f1", width=2)))
    fig.update_layout(title=f"{ticker} — fetched prices", xaxis_title="Time (UTC)",
                      yaxis_title="Price (local currency)", height=height,
                      template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics(pipeline):
    m = pipeline.get_performance_metrics()
    if not m:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Total runs", f"{m['total_runs']}")
    c2.metric("Avg time", f"{m['avg_computation_time']:.1f}s")
    c3.metric("Sim vs hist variance gap", f"{m['avg_variance_reduction'] * 100:.1f}%")


def _risk_frame(tickers: List[str], risk_metrics: Dict) -> pd.DataFrame:
    rows = []
    for t in tickers:
        r = risk_metrics.get(t)
        if not r:
            continue
        rows.append({
            "Asset": t,
            "Volatility": safe_float(r.get("volatility")) * 100,
            "Drift": safe_float(r.get("expected_return")) * 100,
            "Sharpe": safe_float(r.get("sharpe")),
            "VaR": safe_float(r.get("var_95")) * 100,
            "AvgDD": safe_float(r.get("max_drawdown")) * 100,
        })
    return pd.DataFrame(rows)


def render_risk_charts(tickers: List[str], risk_metrics: Dict):
    df = _risk_frame(tickers, risk_metrics)
    if df.empty:
        return
    var_abs = df["VaR"].abs().fillna(0)
    sizes = 12 + 30 * var_abs / (var_abs.max() if var_abs.max() > 0 else 1)  # always positive

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Vol vs drift (bubble = daily VaR)", "Daily VaR 95%", "Avg drawdown per path"))
    fig.add_trace(go.Scatter(
        x=df["Volatility"], y=df["Drift"], mode="markers+text", text=df["Asset"],
        textposition="top center",
        marker=dict(size=sizes, color=df["Sharpe"], colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="Sharpe", x=0.3)),
        name="Assets"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["Asset"], y=df["VaR"], marker_color="#ef4444", name="Daily VaR"), row=1, col=2)
    fig.add_trace(go.Bar(x=df["Asset"], y=df["AvgDD"], marker_color="#f59e0b", name="Avg drawdown"), row=1, col=3)
    fig.update_xaxes(title_text="Volatility % (ann.)", row=1, col=1)
    fig.update_yaxes(title_text="Drift % (ann.)", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=2)
    fig.update_yaxes(title_text="%", row=1, col=3)
    fig.update_layout(height=420, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def screen_signal(metrics: Dict) -> Tuple[str, str]:
    sharpe = safe_float(metrics.get("sharpe"))
    var = safe_float(metrics.get("var_95"))
    dd = safe_float(metrics.get("max_drawdown"))
    if not np.isfinite(sharpe):
        return "⚪ n/a", "missing Sharpe"
    v = var if np.isfinite(var) else -1.0
    if np.isfinite(dd) and dd < -0.25:
        return "🔴 Weak", "deep average drawdown"
    if sharpe > 1.5 and v > -0.02:
        return "🟢 Strong", "high Sharpe, contained daily VaR"
    if sharpe > 0.8 and v > -0.03:
        return "🟡 Positive", "good Sharpe"
    if sharpe > 0.2 and v > -0.04:
        return "🟠 Neutral", "moderate Sharpe"
    return "🔴 Weak", "low Sharpe or large daily VaR"


def render_signals(tickers: List[str], risk_metrics: Dict):
    rows = []
    for t in tickers:
        m = risk_metrics.get(t)
        if not m:
            continue
        signal, reason = screen_signal(m)
        rows.append({"Asset": t, "Screen": signal, "Why": reason,
                     "Sharpe": f"{safe_float(m.get('sharpe')):.2f}",
                     "Daily VaR 95%": f"{safe_float(m.get('var_95')) * 100:.2f}%"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Mechanical screen on simulated Sharpe / VaR / drawdown. The simulated drift comes from "
                   "recent history, so this mostly reflects past performance. Not a recommendation.")


class LiveDashboard:
    """Backward-compatible wrapper. Uses real fetched prices (cached by the fetcher)."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.history: Dict[str, list] = {}
        self.last_update = None
        self.is_running = False

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def refresh(self, tickers: List[str], force: bool = False) -> Dict[str, float]:
        prices = self.pipeline.data_fetcher.get_current_prices(list(tickers), force_refresh=force)
        now = datetime.now(timezone.utc)
        self.last_update = now
        for t, p in prices.items():
            h = self.history.setdefault(t, [])
            h.append({"time": now, "price": p})
            del h[:-500]
        return prices

    def render_price_ticker(self, tickers: List[str]):
        prices = self.refresh(tickers)
        if not prices:
            st.info("No prices available — check API keys or refresh")
            return
        render_price_cards(tickers, prices, self.history, self.pipeline.data_fetcher.price_sources)

    def render_price_chart(self, ticker: str, height: int = 400):
        render_price_chart(ticker, self.history.get(ticker, []), height)

    def render_performance_metrics(self):
        render_performance_metrics(self.pipeline)

    def render_heatmap(self, tickers: List[str], risk_metrics: Dict):
        render_risk_charts(tickers, risk_metrics)

    def render_recommendations(self, risk_metrics: Dict):
        render_signals(list(risk_metrics), risk_metrics)