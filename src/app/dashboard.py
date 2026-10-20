"""
AI Monte Carlo Simulator — Streamlit dashboard (dark theme)
"""

import html
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=False)  # never let a local .env override cloud secrets

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline import MonteCarloPipeline
from src.pipeline.data_fetcher import get_data_fetcher
from src.pipeline.portfolio_optimizer import PortfolioOptimizer
from src.market_utils import currency_symbol, format_price
from src.app.utils import parse_tickers, hex_to_rgba, safe_float
from src.app.live_dashboard import COLORS, render_price_cards, render_risk_charts, render_signals

try:
    from src.explainer.ollama_wrapper import LlamaExplainer
    _EXPLAINER_ERROR = None
except Exception as e:  # keep the app usable without the explainer
    LlamaExplainer = None
    _EXPLAINER_ERROR = str(e)

GAN_CHECKPOINT = os.path.join(project_root, "data", "models", "dfmgan.pth")
MAX_TICKERS = 10
PRESETS = {
    "🇺🇸 Tech": "AAPL, MSFT, GOOGL, NVDA, META",
    "🇮🇳 NSE": "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS",
    "🇬🇧 UK": "BP.L, HSBA.L, GSK.L, AZN.L, DGE.L",
    "₿ Crypto": "BTC-USD, ETH-USD, BNB-USD, SOL-USD",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
* { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
.main-header { background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 1.6rem 2rem;
  border-radius: 24px; color: white; margin-bottom: 1.5rem; box-shadow: 0 20px 40px rgba(99,102,241,0.3); }
.pill { background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.85rem; }
.metric-card { background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 16px; padding: 1rem;
  color: white; text-align: center; box-shadow: 0 10px 30px rgba(99,102,241,0.3); }
.metric-value { font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0; word-break: break-word; }
.metric-label { font-size: 0.8rem; opacity: 0.9; }
.section-header { background: linear-gradient(90deg, #6366f1 0%, transparent 100%); padding: 0.9rem 1.6rem;
  border-radius: 12px; margin: 1.8rem 0 1.2rem 0; color: white; font-weight: 600; font-size: 1.2rem; }
.stButton > button { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; border: none;
  border-radius: 12px; padding: 0.5rem 1rem; font-weight: 600; }
</style>
"""


# ============================================================
# STATE
# ============================================================
def _defaults() -> Dict:
    return {
        "pipeline": None,
        "results": None,
        "results_request": None,
        "explanation": None,
        "explanation_note": None,
        "stocks_input": PRESETS["🇺🇸 Tech"],
        "live_prices": {},
        "prices_for": None,
        "price_history": {},
        "force_price_refresh": False,
    }


def _init_state():
    for k, v in _defaults().items():
        if k not in st.session_state:
            st.session_state[k] = v


def _set_preset(value: str):
    st.session_state.stocks_input = value


def _request_refresh():
    fetcher = get_data_fetcher()
    fetcher.cache_timestamp.clear()        # current prices
    fetcher.historical_timestamp.clear()   # history
    _portfolio_analysis.clear()
    st.session_state.force_price_refresh = True


# ============================================================
# DATA
# ============================================================
def _update_prices(fetcher, stocks: List[str]):
    """Fetch only when the ticker set changes or Refresh was pressed (respects the fetcher cache)."""
    force = bool(st.session_state.force_price_refresh)
    key = tuple(stocks)
    if not stocks or (st.session_state.prices_for == key and not force):
        return
    st.session_state.force_price_refresh = False
    with st.spinner("Fetching prices..."):
        try:
            prices = fetcher.get_current_prices(stocks, force_refresh=force)
        except Exception as e:
            st.error(f"Price fetch failed: {e}")
            prices = {}
    st.session_state.live_prices = {t: p for t, p in prices.items() if p and p > 0}
    st.session_state.prices_for = key
    now = datetime.now(timezone.utc)
    for t, p in st.session_state.live_prices.items():
        h = st.session_state.price_history.setdefault(t, [])
        h.append({"time": now, "price": p})
        del h[:-100]


def _source_summary(fetcher, tickers: List[str]) -> str:
    counts: Dict[str, int] = {}
    for t in tickers:
        s = fetcher.price_sources.get(t)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return ", ".join(f"{s} ×{n}" for s, n in counts.items()) or "no price source yet"


@st.cache_data(ttl=1800, show_spinner=False)
def _portfolio_analysis(requested: Tuple[str, ...], period: str, used: Tuple[str, ...]) -> Dict:
    """Mean/covariance from real history + optimiser output. Raises (not cached) on failure."""
    from sklearn.covariance import LedoitWolf

    hist = get_data_fetcher().get_historical_data(requested, period)
    if hist is None:
        raise ValueError("historical data unavailable")
    missing = [t for t in used if t not in hist.columns]
    if missing:
        raise ValueError(f"no history for {missing}")
    px = hist[list(used)]
    rets = np.log(px / px.shift(1)).dropna()
    if len(rets) < 30:
        raise ValueError("not enough overlapping history")

    mu = rets.mean().values * 252
    cov = LedoitWolf().fit(rets.values).covariance_ * 252
    opt = PortfolioOptimizer(mu, cov, asset_names=list(used))
    return {
        "mu": mu, "cov": cov, "n_obs": len(rets),
        "max_sharpe": opt.optimize_max_sharpe(0.02),
        "min_vol": opt.optimize_min_volatility(),
        "frontier": opt.generate_efficient_frontier(30),
    }


def _run_simulation(pipeline, stocks, period, n_sims, filter_pct, use_gan) -> Dict:
    pipeline.n_simulations = n_sims
    pipeline.filter_top_k = filter_pct
    pipeline.use_gan = use_gan
    pipeline.n_assets = len(stocks)

    progress, status = st.progress(0), st.empty()
    state = {"pct": 0, "msg": "Starting..."}
    holder: Dict = {}

    def _cb(pct, msg):
        state["pct"], state["msg"] = pct, msg

    def _work():
        try:
            holder["results"] = pipeline.run_simulation(
                tickers=list(stocks), period=period, use_real_options=False, progress_callback=_cb)
        except Exception as ex:
            holder["error"] = ex

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    while t.is_alive():
        progress.progress(int(min(state["pct"], 99)))
        status.text(state["msg"])
        time.sleep(0.15)
    progress.empty()
    status.empty()
    if "error" in holder:
        raise holder["error"]
    return holder["results"]


# ============================================================
# RESULT SECTIONS
# ============================================================
def _render_stats(results: Dict, req: Dict):
    meta = results.get("metadata", {})
    used = meta.get("tickers", [])
    cards = [
        ("🔄", "Paths", f"{meta.get('n_simulations', 0):,}"),
        ("📊", "Assets", f"{len(used)}/{len(req['tickers'])}"),
        ("📅", "Horizon", f"{results.get('horizon_days', '—')} days"),
        ("⏱️", "Time", f"{meta.get('computation_time', 0):.1f}s"),
        ("📡", "Data", str(meta.get("data_source", "—"))),
        ("🤖", "Engine", "GAN" if meta.get("use_gan") else "GBM + jumps"),
    ]
    for col, (emoji, label, value) in zip(st.columns(len(cards)), cards):
        col.markdown(f"""<div class="metric-card"><div style="font-size:1.3rem;">{emoji}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)


def _render_forecast(results: Dict, used: List[str]):
    start = results.get("start_prices", {})
    cur = results.get("current_prices", {})
    exp = results.get("expected_prices", {})
    ci = results.get("confidence_intervals", {})
    dists = results.get("price_distributions", {})
    rows = []
    for t in used:
        s = safe_float(start.get(t, cur.get(t)))
        e = safe_float(exp.get(t))
        lo, hi = (list(ci.get(t) or []) + [np.nan, np.nan])[:2]
        d = np.asarray(dists.get(t, []), float)
        p_up = float(np.mean(d > s)) if d.size and np.isfinite(s) else np.nan
        rows.append({
            "Asset": t,
            "Start": format_price(s, t),
            "Forecast (mean)": format_price(e, t),
            "Change": f"{(e / s - 1) * 100:+.1f}%" if np.isfinite(s) and s > 0 and np.isfinite(e) else "—",
            "95% range": f"{format_price(lo, t)} – {format_price(hi, t)}",
            "P(up)": f"{p_up * 100:.0f}%" if np.isfinite(p_up) else "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Mean of all simulated paths after {results.get('horizon_days', '?')} trading days. "
                   "Prices in local currency (LSE quotes are in pence).")


def _render_paths_tab(results: Dict, used: List[str]):
    selected = st.multiselect("Assets", options=used, default=used[:3], key="paths_assets")
    if not selected:
        return
    normalise = st.checkbox("Normalise to 100 at start", value=len(selected) > 1, key="paths_norm")
    pct = results.get("path_percentiles") or {}
    start = results.get("start_prices", {})
    sample = np.asarray(results.get("path_sample") or [], float)

    fig = go.Figure()
    for i, t in enumerate(selected):
        color = COLORS[i % len(COLORS)]
        if t in pct:
            b = {k: np.asarray(v, float) for k, v in pct[t].items()}
            base = safe_float(start.get(t), b["50"][0])
            scale = 100.0 / base if normalise and base > 0 else 1.0
            x = list(range(len(b["50"])))
            fig.add_trace(go.Scatter(x=x, y=b["95"] * scale, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=b["5"] * scale, line=dict(width=0), fill="tonexty",
                                     fillcolor=hex_to_rgba(color, 0.12), name=f"{t} 90% band"))
            fig.add_trace(go.Scatter(x=x, y=b["75"] * scale, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=b["25"] * scale, line=dict(width=0), fill="tonexty",
                                     fillcolor=hex_to_rgba(color, 0.28), name=f"{t} 50% band"))
            fig.add_trace(go.Scatter(x=x, y=b["50"] * scale, name=f"{t} median", line=dict(color=color, width=3)))
        elif sample.ndim == 3 and used.index(t) < sample.shape[2]:  # old results without percentiles
            j = used.index(t)
            base = sample[0, 0, j]
            scale = 100.0 / base if normalise and base > 0 else 1.0
            for k in range(sample.shape[0]):
                fig.add_trace(go.Scatter(y=sample[k, :, j] * scale, line=dict(color=color, width=1),
                                         opacity=0.5, showlegend=k == 0, name=f"{t} sample paths"))

    currencies = {currency_symbol(t) for t in selected}
    y_title = ("Index (start = 100)" if normalise
               else f"Price ({currencies.pop().strip()})" if len(currencies) == 1 else "Price (local currency)")
    fig.update_layout(title="Simulated price paths (bands from all paths)", xaxis_title="Trading days",
                      yaxis_title=y_title, height=520, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def _render_distribution_tab(results: Dict, used: List[str]):
    t = st.selectbox("Asset", options=used, key="dist_asset")
    d = np.asarray(results.get("price_distributions", {}).get(t, []), float)
    if d.size == 0:
        st.info("No distribution available")
        return
    s = safe_float(results.get("start_prices", {}).get(t))
    fig = go.Figure(go.Histogram(x=d, nbinsx=60, marker_color="#6366f1"))
    if np.isfinite(s):
        fig.add_vline(x=s, line_dash="dash", line_color="#f59e0b", annotation_text="Start")
    fig.add_vline(x=float(d.mean()), line_dash="dot", line_color="#10b981", annotation_text="Mean")
    fig.update_layout(title=f"{t}: final price across {d.size:,} paths", height=420, template="plotly_dark",
                      xaxis_title=f"Price ({currency_symbol(t).strip() or 'points'})", yaxis_title="Paths")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(loss)", f"{np.mean(d < s) * 100:.0f}%" if np.isfinite(s) else "—")
    c2.metric("5th percentile", format_price(np.percentile(d, 5), t))
    c3.metric("Median", format_price(np.median(d), t))
    c4.metric("95th percentile", format_price(np.percentile(d, 95), t))


def _render_risk_tab(results: Dict, used: List[str]):
    risk = results.get("risk_metrics", {})
    rows = []
    for t in used:
        r = risk.get(t)
        if not r:
            continue
        rows.append({
            "Asset": t,
            "Daily VaR 95%": f"{safe_float(r.get('var_95')) * 100:.2f}%",
            "Daily CVaR 95%": f"{safe_float(r.get('cvar_95')) * 100:.2f}%",
            "Sharpe": f"{safe_float(r.get('sharpe')):.2f}",
            "Sortino": f"{safe_float(r.get('sortino')):.2f}",
            "Volatility (ann.)": f"{safe_float(r.get('volatility')) * 100:.1f}%",
            "Drift (ann.)": f"{safe_float(r.get('expected_return')) * 100:.1f}%",
            "Avg drawdown": f"{safe_float(r.get('max_drawdown')) * 100:.1f}%",
            "Worst drawdown": f"{safe_float(r.get('worst_drawdown')) * 100:.1f}%",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    render_risk_charts(used, risk)
    render_signals(used, risk)


def _render_portfolio_tab(req: Dict, used: List[str]):
    if len(used) < 2:
        st.info("Need at least 2 assets with data.")
        return
    try:
        with st.spinner("Estimating covariance from history..."):
            pa = _portfolio_analysis(tuple(req["tickers"]), req["period"], tuple(used))
    except Exception as e:
        st.info(f"Portfolio view unavailable: {e}")
        return

    mu, cov = pa["mu"], pa["cov"]
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(len(used)), size=2000)
    r = W @ mu
    v = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v * 100, y=r * 100, mode="markers", name="Random portfolios",
                             marker=dict(size=4, color=(r - 0.02) / (v + 1e-12), colorscale="RdYlGn",
                                         showscale=True, colorbar=dict(title="Sharpe"))))
    fr = pa["frontier"]
    if not fr.empty:
        fig.add_trace(go.Scatter(x=fr["volatility"] * 100, y=fr["return"] * 100, mode="lines",
                                 name="Efficient frontier", line=dict(color="white", width=2)))
    for label, sol, sym in (("Max Sharpe", pa["max_sharpe"], "star"), ("Min volatility", pa["min_vol"], "diamond")):
        fig.add_trace(go.Scatter(x=[sol["volatility"] * 100], y=[sol["return"] * 100], mode="markers",
                                 name=label, marker=dict(size=16, symbol=sym)))
    fig.update_layout(title="Efficient frontier (historical mean, Ledoit-Wolf covariance)",
                      xaxis_title="Volatility % (ann.)", yaxis_title="Expected return % (ann.)",
                      height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "Asset": used,
        "Max-Sharpe weight": [f"{w * 100:.1f}%" for w in pa["max_sharpe"]["weights"]],
        "Min-vol weight": [f"{w * 100:.1f}%" for w in pa["min_vol"]["weights"]],
    }), use_container_width=True, hide_index=True)
    st.caption(f"Estimated from {pa['n_obs']} overlapping daily returns. Returns are in each asset's local "
               "currency (no FX). Max-Sharpe weights are very sensitive to noisy return estimates.")


def _render_ai(results: Dict, used: List[str]):
    c1, c2 = st.columns([1, 3])
    with c1:
        question = st.text_input("Question (optional)", key="ai_question")
        if st.button("Generate insights", use_container_width=True):
            if LlamaExplainer is None:
                st.session_state.explanation = None
                st.session_state.explanation_note = f"Explainer import failed: {_EXPLAINER_ERROR}"
            else:
                with st.spinner("Analysing..."):
                    ex = LlamaExplainer()
                    text = ex.explain_simulation_results(
                        tickers=used,
                        expected_prices=results.get("expected_prices", {}),
                        confidence_intervals=results.get("confidence_intervals", {}),
                        risk_metrics=results.get("risk_metrics", {}),
                        variance_reduction=results.get("variance_reduction", 0.0),
                        user_question=question or None,
                        horizon_days=results.get("horizon_days"),
                    )
                st.session_state.explanation = text
                st.session_state.explanation_note = (
                    f"Generated by {ex.model_name}" if ex.last_source == "llm"
                    else "Rule-based fallback (the AI model was not reached)")
    with c2:
        if st.session_state.explanation_note:
            st.caption(st.session_state.explanation_note)
        if st.session_state.explanation:
            with st.container():
                st.markdown(st.session_state.explanation)  # plain Markdown: no raw HTML from the model
        else:
            st.info('Click "Generate insights" for an AI summary of these results.')


def _render_downloads(results: Dict, used: List[str]):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📥 JSON", json.dumps(results, default=str), "results.json", use_container_width=True)
    with c2:
        ci = results.get("confidence_intervals", {})
        risk = results.get("risk_metrics", {})
        df = pd.DataFrame([{
            "Asset": t,
            "Start": results.get("start_prices", {}).get(t),
            "Forecast": results.get("expected_prices", {}).get(t),
            "CI_low": (ci.get(t) or [None, None])[0],
            "CI_high": (ci.get(t) or [None, None])[1],
            "Daily_VaR_95": (risk.get(t) or {}).get("var_95"),
            "Sharpe": (risk.get(t) or {}).get("sharpe"),
        } for t in used])
        st.download_button("📊 CSV", df.to_csv(index=False), "forecast.csv", use_container_width=True)
    with c3:
        if st.button("🔄 Clear results", use_container_width=True):
            st.session_state.results = None
            st.session_state.results_request = None
            st.session_state.explanation = None
            st.session_state.explanation_note = None


def _render_results(fetcher, stocks: List[str], use_gan_requested: bool):
    results = st.session_state.results
    req = st.session_state.results_request or {"tickers": stocks, "period": "2y"}
    meta = results.get("metadata", {})
    used = meta.get("tickers", [])

    if req["tickers"] != stocks:
        st.info(f"Showing results for {', '.join(req['tickers'])}. Inputs changed — run again to update.")
    missing = meta.get("missing_tickers", [])
    if missing:
        st.warning(f"No market data for: {', '.join(missing)}. They were left out of the simulation.")
        with st.expander("Provider errors"):
            st.json({t: fetcher.last_errors.get(t, "no error recorded") for t in missing})
    if use_gan_requested and not meta.get("use_gan"):
        st.info("AI generator was requested but no compatible trained checkpoint loaded, so GBM was used.")
    if not used:
        return

    _render_stats(results, req)
    st.markdown('<div class="section-header">💰 Price Forecast</div>', unsafe_allow_html=True)
    _render_forecast(results, used)

    st.markdown('<div class="section-header">📈 Analytics</div>', unsafe_allow_html=True)
    tabs = st.tabs(["📊 Price Paths", "📉 Distribution", "🔮 Risk", "📊 Portfolio", "📋 Options"])
    with tabs[0]:
        _render_paths_tab(results, used)
    with tabs[1]:
        _render_distribution_tab(results, used)
    with tabs[2]:
        _render_risk_tab(results, used)
    with tabs[3]:
        _render_portfolio_tab(req, used)
    with tabs[4]:
        st.info("Option chains need a paid data provider and are not available in this build.")

    st.markdown('<div class="section-header">🧠 AI Market Analysis</div>', unsafe_allow_html=True)
    _render_ai(results, used)
    st.markdown("---")
    _render_downloads(results, used)


# ============================================================
# MAIN
# ============================================================
def run_dashboard():
    st.set_page_config(page_title="AI Monte Carlo Simulator Pro", page_icon="📈",
                       layout="wide", initial_sidebar_state="expanded")
    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    fetcher = get_data_fetcher()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.markdown("**Quick select**")
        names = list(PRESETS)
        for row in range(0, len(names), 2):
            cols = st.columns(2)
            for col, name in zip(cols, names[row:row + 2]):
                col.button(name, on_click=_set_preset, args=(PRESETS[name],), use_container_width=True)

        raw = st.text_area("**Symbols** (comma or newline separated)", key="stocks_input", height=80)
        stocks, invalid, dropped = parse_tickers(raw, MAX_TICKERS)
        if invalid:
            st.warning(f"Ignored invalid symbols: {', '.join(invalid[:5])}")
        if dropped:
            st.warning(f"Max {MAX_TICKERS} symbols (free API limits). Ignored: {', '.join(dropped)}")

        st.markdown("---")
        st.markdown("**Simulation**")
        n_sims = st.slider("Number of paths", 100, 5000, 1000, 100)
        has_ckpt = os.path.exists(GAN_CHECKPOINT)
        use_gan = st.checkbox("🤖 AI generator (GAN)", value=False,
                              help="Uses data/models/dfmgan.pth when present and trained for this many assets.")
        if use_gan and not has_ckpt:
            st.caption("No trained checkpoint found, so GBM will be used.")
        filter_pct = st.slider("Display sample (top % most typical paths)", 1, 50, 10) / 100
        period = st.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=2)

        run_button = st.button("🚀 RUN SIMULATION", use_container_width=True, type="primary")
        st.button("🔄 Refresh data", on_click=_request_refresh, use_container_width=True)
        if st.button("🧪 Test AI connection", use_container_width=True):
            if LlamaExplainer is None:
                st.error(f"Explainer import failed: {_EXPLAINER_ERROR}")
            else:
                ok, msg = LlamaExplainer().check_connection()
                (st.success if ok else st.error)(msg)
        st.caption(f"Tracking {len(stocks)} assets")

    # ---------------- PRICES ----------------
    _update_prices(fetcher, stocks)

    now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")
    st.markdown(f"""
    <div class="main-header">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.8rem;">
        <div>
          <h1 style="margin:0;font-size:2.2rem;">🚀 AI Monte Carlo Pro</h1>
          <p style="margin:0.4rem 0 0 0;opacity:0.9;">Portfolio simulation & risk analytics</p>
        </div>
        <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
          <span class="pill">📡 {html.escape(_source_summary(fetcher, stocks))}</span>
          <span class="pill">{now_utc}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.pipeline is None:
        st.session_state.pipeline = MonteCarloPipeline(
            n_assets=max(len(stocks), 1), n_simulations=n_sims, filter_top_k=filter_pct,
            use_gan=use_gan, use_live_data=False)
    pipeline = st.session_state.pipeline
    pipeline.gan_checkpoint = GAN_CHECKPOINT if has_ckpt else None

    st.markdown("### 📊 Market Prices")
    live = st.session_state.live_prices
    if live:
        render_price_cards(stocks, live, st.session_state.price_history, fetcher.price_sources)
    no_price = [t for t in stocks if t not in live]
    if no_price:
        st.warning(f"No current price for: {', '.join(no_price)}. Press Refresh to retry.")
        with st.expander("Provider errors"):
            st.json({t: fetcher.last_errors.get(t, "no error recorded") for t in no_price})

    # ---------------- RUN ----------------
    if run_button:
        if not stocks:
            st.warning("Enter at least one symbol.")
        else:
            try:
                results = _run_simulation(pipeline, stocks, period, n_sims, filter_pct, use_gan)
                st.session_state.results = results
                st.session_state.results_request = {"tickers": list(stocks), "period": period}
                st.session_state.explanation = None
                st.session_state.explanation_note = None
                st.success("✅ Simulation complete")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    # ---------------- RESULTS ----------------
    if st.session_state.results:
        _render_results(fetcher, stocks, use_gan)
    else:
        st.markdown("""<div style="text-align:center;padding:3rem;">
            <h2 style="font-size:2.5rem;">🚀 Ready</h2>
            <p style="font-size:1.1rem;color:#94a3b8;">Pick symbols in the sidebar and run a simulation</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#64748b;padding:1rem;font-size:0.8rem;">
        <p style="color:#ef4444;font-weight:bold;">⚠️ NOT FINANCIAL ADVICE</p>
        <p>Educational and research purposes only.</p></div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    run_dashboard()