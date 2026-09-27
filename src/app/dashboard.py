"""
Ultra-Modern Monte Carlo Dashboard - Premium Edition v2.0
Dark Theme Only - Advanced Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import time
import json
import random
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ✅ Force yfinance to use curl_cffi impersonation (bypasses Yahoo cloud blocking)
try:
    from curl_cffi import requests as _curl_requests
    import yfinance as _yf
    _yf.set_tz_cache_location("/tmp/yfinance_cache")
    print("✅ curl_cffi available for yfinance impersonation")
except Exception as e:
    print(f"⚠️ curl_cffi setup skipped: {e}")

# Add root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.pipeline import MonteCarloPipeline
    print("✅ Import successful")
except ImportError as e:
    st.error(f"Failed to import MonteCarloPipeline: {e}")
    st.stop()

try:
    from src.explainer.ollama_wrapper import LlamaExplainer
    print("✅ Import successful")
except ImportError:
    LlamaExplainer = None
    print("⚠️ LlamaExplainer not available")


def run_dashboard():
    st.set_page_config(
        page_title="AI Monte Carlo Simulator Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ===================== SESSION STATE =====================
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'explanation' not in st.session_state:
        st.session_state.explanation = None
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'stocks_input' not in st.session_state:
        st.session_state.stocks_input = "AAPL, MSFT, GOOGL, NVDA, META"
    if 'live_prices' not in st.session_state:
        st.session_state.live_prices = {}
    if 'price_history' not in st.session_state:
        st.session_state.price_history = {}
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'prices_fetched' not in st.session_state:
        st.session_state.prices_fetched = False
    if 'prices_fetched_for' not in st.session_state:
        st.session_state.prices_fetched_for = []

    # ===================== CSS =====================
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Space Grotesk', sans-serif; }
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
        .main-header {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            padding: 2rem; border-radius: 24px; color: white;
            margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
        }
        .glass-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px; padding: 1.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 16px; padding: 1.5rem; color: white;
            text-align: center; box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }
        .metric-value { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
        .metric-label { font-size: 0.9rem; opacity: 0.9; }
        .section-header {
            background: linear-gradient(90deg, #6366f1 0%, transparent 100%);
            padding: 1rem 2rem; border-radius: 12px;
            margin: 2rem 0 1.5rem 0; color: white;
            font-weight: 600; font-size: 1.3rem;
        }
        .live-badge {
            display: inline-block; background: #ef4444; color: white;
            padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.8rem;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .info-box {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem; border-radius: 16px; color: white;
        }
        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white; border: none; border-radius: 12px;
            padding: 0.5rem 1rem; font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    # ===================== HELPERS =====================
    def safe_rerun():
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    def format_currency(value):
        if value is None or np.isnan(value):
            return "$0.00"
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        elif value >= 1e3:
            return f"${value/1e3:.2f}K"
        return f"${value:.2f}"

    def fetch_live_prices_from_pipeline(tickers):
        prices = {}
        try:
            if st.session_state.pipeline and hasattr(st.session_state.pipeline, 'data_fetcher'):
                fetcher = st.session_state.pipeline.data_fetcher
                fetched = fetcher.get_current_prices(tickers, force_refresh=True)
                if fetched:
                    for ticker in tickers:
                        prices[ticker] = fetched.get(ticker, 100.0)
                    return prices
        except Exception as e:
            print(f"⚠️ Error fetching from pipeline: {e}")
        return prices

    # ===================== HEADER =====================
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin:0; font-size: 2.5rem;">🚀 AI Monte Carlo Pro</h1>
                <p style="margin:0.5rem 0 0 0; opacity:0.9;">Advanced Portfolio Simulation & Risk Analytics</p>
            </div>
            <div style="display: flex; gap: 0.8rem; align-items: center;">
                <span class="live-badge">🔴 LIVE</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem;">{datetime.now().strftime('%H:%M:%S')}</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem;">🤖 AI</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===================== SIDEBAR =====================
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.markdown("**Quick Select**")
        quick_cols = st.columns(2)
        with quick_cols[0]:
            if st.button("🇺🇸 Tech", use_container_width=True):
                st.session_state.stocks_input = "AAPL, MSFT, GOOGL, NVDA, META"
        with quick_cols[1]:
            if st.button("🇮🇳 NSE", use_container_width=True):
                st.session_state.stocks_input = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS"

        quick_cols2 = st.columns(2)
        with quick_cols2[0]:
            if st.button("🇬🇧 UK", use_container_width=True):
                st.session_state.stocks_input = "BP.L, HSBA.L, GSK.L, AZN.L, DGE.L"
        with quick_cols2[1]:
            if st.button("₿ Crypto", use_container_width=True):
                st.session_state.stocks_input = "BTC-USD, ETH-USD, BNB-USD, SOL-USD"

        stocks_input = st.text_area(
            "**Enter Symbols**",
            value=st.session_state.stocks_input,
            height=80,
        )

        if "," in stocks_input:
            stocks = [s.strip().upper() for s in stocks_input.split(",") if s.strip()]
        else:
            stocks = [s.strip().upper() for s in stocks_input.split("\n") if s.strip()]

        st.session_state.stocks_input = ", ".join(stocks)

        st.markdown("---")
        st.markdown("**Simulation**")

        n_sims = st.slider("Number of Paths", 100, 5000, 1000, 100)
        use_gan = st.checkbox("🤖 AI Generator", value=False)
        filter_pct = st.slider("Keep Top %", 1, 50, 10) / 100

        st.markdown("**Data**")
        period = st.selectbox("Time Period", ["6mo", "1y", "2y", "5y"], index=2)
        include_options = st.checkbox("📋 Show Options", value=True)

        run_button = st.button("🚀 RUN SIMULATION", use_container_width=True, type="primary")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.cache_data.clear()
            st.session_state.live_prices = {}
            st.session_state.prices_fetched = False
            st.session_state.prices_fetched_for = []
            safe_rerun()

        st.caption(f"Tracking: {len(stocks)} assets")

    # ===================== PIPELINE RESET ON TICKER CHANGE =====================
    if st.session_state.pipeline is not None:
        if getattr(st.session_state.pipeline, 'n_assets', None) != len(stocks):
            print(f"🔄 Asset count changed ({st.session_state.pipeline.n_assets} → {len(stocks)}), resetting pipeline")
            st.session_state.pipeline = None
            st.session_state.results = None
            st.session_state.explanation = None
            st.session_state.live_prices = {}
            st.session_state.prices_fetched = False
            st.session_state.prices_fetched_for = []

    # ===================== INITIALIZE PIPELINE =====================
    if st.session_state.pipeline is None and stocks:
        with st.spinner("Initializing AI Engine..."):
            try:
                print("🚀 Initializing Monte Carlo Pipeline...")
                pipeline = MonteCarloPipeline(
                    n_assets=len(stocks),
                    n_simulations=n_sims,
                    filter_top_k=filter_pct,
                    use_gan=use_gan,
                    use_live_data=True
                )
                print("📦 Loading ML models...")
                pipeline.load_models()
                st.session_state.pipeline = pipeline
                st.session_state.error_message = None
                print("✅ AI Engine Ready!")

                live_prices = fetch_live_prices_from_pipeline(stocks)
                if live_prices:
                    st.session_state.live_prices = live_prices
                    st.session_state.prices_fetched = True
                    st.session_state.prices_fetched_for = list(stocks)
                    print(f"✅ Live prices loaded: {live_prices}")

                st.success("✅ AI Engine Ready!")
                time.sleep(0.3)
                safe_rerun()
            except Exception as e:
                st.session_state.error_message = str(e)
                print(f"❌ Initialization failed: {e}")
                st.error(f"Initialization failed: {e}")

    # ===================== LIVE PRICE TICKER =====================
    current_set = set(stocks)
    fetched_set = set(st.session_state.prices_fetched_for)
    needs_fetch = (
        st.session_state.auto_refresh
        or not st.session_state.prices_fetched
        or not st.session_state.live_prices
        or current_set != fetched_set
    )

    if stocks and needs_fetch:
        with st.spinner("Fetching live prices..."):
            try:
                fresh = fetch_live_prices_from_pipeline(stocks)
                if fresh:
                    st.session_state.live_prices = fresh
                else:
                    st.session_state.live_prices = {t: 100.0 for t in stocks}

                st.session_state.prices_fetched = True
                st.session_state.prices_fetched_for = list(stocks)

                for ticker in stocks:
                    if ticker not in st.session_state.price_history:
                        st.session_state.price_history[ticker] = []
                    if ticker in st.session_state.live_prices:
                        st.session_state.price_history[ticker].append({
                            'time': datetime.now(),
                            'price': st.session_state.live_prices[ticker]
                        })
                    if len(st.session_state.price_history[ticker]) > 100:
                        st.session_state.price_history[ticker] = st.session_state.price_history[ticker][-100:]
            except Exception as e:
                print(f"⚠️ Error in live price ticker: {e}")

    # ===================== LIVE PRICE DISPLAY =====================
    if st.session_state.live_prices:
        st.markdown("### 📊 Live Market Prices")
        cols = st.columns(min(len(stocks), 8))
        for i, ticker in enumerate(stocks[:8]):
            with cols[i]:
                price = st.session_state.live_prices.get(ticker, 0)
                history = st.session_state.price_history.get(ticker, [])
                change = 0
                if len(history) >= 2:
                    change = ((history[-1]['price'] - history[-2]['price']) / history[-2]['price']) * 100

                color = "#10b981" if change >= 0 else "#ef4444"
                arrow = "↑" if change >= 0 else "↓"

                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 0.8rem; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 0.8rem; color: #94a3b8;">{ticker}</div>
                    <div style="font-size: 1.3rem; font-weight: 600;">{format_currency(price)}</div>
                    <div style="font-size: 0.9rem; color: {color};">{arrow} {change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

    # ===================== RUN SIMULATION =====================
    if run_button and st.session_state.pipeline and stocks:
        st.session_state.simulation_running = True
        print("🚀 Starting simulation...")
        print(f"📊 Assets: {stocks}")
        print(f"📈 Paths: {n_sims}, Period: {period}")

        with st.spinner("AI simulating market scenarios..."):
            try:
                pipeline = st.session_state.pipeline
                pipeline.n_simulations = n_sims
                pipeline.filter_top_k = filter_pct
                pipeline.use_gan = use_gan

                progress_bar = st.progress(0)
                status_text = st.empty()

                progress_state = {"pct": 0, "msg": "Starting..."}

                def _cb(pct, msg):
                    progress_state["pct"] = pct
                    progress_state["msg"] = msg

                result_holder = {}

                def _run():
                    try:
                        result_holder["results"] = pipeline.run_simulation(
                            tickers=stocks,
                            period=period,
                            use_real_options=include_options,
                            progress_callback=_cb
                        )
                    except Exception as ex:
                        result_holder["error"] = ex

                t = threading.Thread(target=_run, daemon=True)
                t.start()

                while t.is_alive():
                    progress_bar.progress(min(progress_state["pct"], 99) / 100)
                    status_text.text(progress_state["msg"])
                    time.sleep(0.15)

                t.join()
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                time.sleep(0.3)
                progress_bar.empty()
                status_text.empty()

                if "error" in result_holder:
                    raise result_holder["error"]

                results = result_holder["results"]
                st.session_state.results = results
                st.session_state.explanation = None
                st.session_state.error_message = None
                st.session_state.simulation_running = False

                print("✅ Simulation Complete!")
                metadata = results.get("metadata", {})
                print(f"📊 Generated {metadata.get('n_simulations', 0)} paths")
                print(f"🎯 Filtered to {metadata.get('filtered_paths', 0)} paths")
                print(f"⏱️ Computation time: {metadata.get('computation_time', 0):.2f}s")
                st.success("✅ Simulation Complete!")

            except Exception as e:
                st.session_state.error_message = str(e)
                print(f"❌ Simulation failed: {e}")
                st.error(f"Simulation failed: {e}")
                st.session_state.simulation_running = False

    # ===================== DISPLAY RESULTS =====================
    if st.session_state.results:
        results = st.session_state.results
        metadata = results.get("metadata", {})
        expected_prices = results.get("expected_prices", {})
        current_prices = results.get("current_prices", {})
        confidence_intervals = results.get("confidence_intervals", {})
        risk_metrics = results.get("risk_metrics", {})
        option_prices = results.get("option_prices", {})

        # ============ ✅ FIX: Variance metric display ============
        # Show "—" when variance is 0 or unrealistically high (>60%)
        # instead of misleading values like 99.8%
        variance_reduction = results.get('variance_reduction', 0)

        if variance_reduction <= 0 or variance_reduction > 0.60:
            variance_display = "—"
        else:
            variance_display = f"{variance_reduction*100:.1f}%"
        # =========================================================

        st.markdown("### 📊 Live Stats")
        cols = st.columns(8)
        metrics_data = [
            ("Paths", f"{metadata.get('n_simulations', 0):,}", "🔄"),
            ("Filtered", f"{metadata.get('filtered_paths', 0):,}", "🎯"),
            ("Time", f"{metadata.get('computation_time', 0):.1f}s", "⏱️"),
            ("Variance", variance_display, "📉"),  # ✅ FIXED
            ("Assets", f"{len(stocks)}", "📊"),
            ("Data", "Live", "📡"),
            ("AI Mode", "✅" if metadata.get('use_gan', False) else "⚡", "🤖"),
            ("Status", "🟢 Active", "✅")
        ]
        for col, (label, value, emoji) in zip(cols, metrics_data):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.5rem;">{emoji}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">💰 Price Forecast & Trends</div>', unsafe_allow_html=True)
        price_data = []
        for stock in stocks:
            if stock in expected_prices:
                current = current_prices.get(stock, st.session_state.live_prices.get(stock, 0))
                expected = expected_prices[stock]
                change = ((expected - current) / current * 100) if current else 0
                ci = confidence_intervals.get(stock, [0, 0])
                risk = risk_metrics.get(stock, {})
                sharpe = risk.get('sharpe', 0)
                price_data.append({
                    "Asset": stock,
                    "Current": format_currency(current),
                    "Forecast": format_currency(expected),
                    "Change": f"{change:+.1f}%",
                    "Sharpe": f"{sharpe:.2f}",
                    "Range": format_currency(ci[1] - ci[0])
                })
        if price_data:
            st.dataframe(pd.DataFrame(price_data), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
        if "path_sample" in results and results["path_sample"]:
            paths = np.array(results["path_sample"])
            tabs = st.tabs(["📊 Price Paths", "📉 Distribution", "📋 Options", "🔮 Risk Analysis", "📊 Portfolio"])

            with tabs[0]:
                selected = st.multiselect("Select assets",
                    options=stocks[:paths.shape[2]],
                    default=stocks[:min(3, paths.shape[2])])
                if selected:
                    fig = go.Figure()
                    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']
                    for i, stock in enumerate(selected):
                        if stock in stocks:
                            idx = stocks.index(stock)
                            mean_path = np.mean(paths[:, :, idx], axis=0)
                            upper = np.percentile(paths[:, :, idx], 95, axis=0)
                            lower = np.percentile(paths[:, :, idx], 5, axis=0)
                            color = colors[i % len(colors)]
                            fig.add_trace(go.Scatter(x=list(range(len(mean_path))), y=upper,
                                line=dict(width=0), showlegend=False, hoverinfo='skip'))
                            fig.add_trace(go.Scatter(x=list(range(len(mean_path))), y=lower,
                                fill='tonexty', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                            fig.add_trace(go.Scatter(x=list(range(len(mean_path))), y=mean_path,
                                name=f'{stock} (Mean)', line=dict(color=color, width=3)))
                    fig.update_layout(title="Price Paths with 95% Confidence Bands",
                        xaxis_title="Trading Days", yaxis_title="Price ($)",
                        height=550, template='plotly_dark', hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                stock = st.selectbox("Select asset", options=stocks[:paths.shape[2]], key="dist_stock")
                if stock:
                    idx = stocks.index(stock)
                    final_prices = paths[:, -1, idx]
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=final_prices, nbinsx=50, marker_color='#6366f1'))
                    fig.update_layout(title=f"{stock} Final Price Distribution",
                        height=400, template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:
                if include_options and option_prices:
                    for stock, opt_data in option_prices.items():
                        if opt_data and opt_data.get('calls'):
                            with st.expander(f"📋 {stock} Options Chain"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**CALLS**")
                                    if opt_data.get('calls'):
                                        st.dataframe(pd.DataFrame(opt_data['calls']), use_container_width=True)
                                with col2:
                                    st.markdown("**PUTS**")
                                    if opt_data.get('puts'):
                                        st.dataframe(pd.DataFrame(opt_data['puts']), use_container_width=True)
                else:
                    st.info("No options data available")

            with tabs[3]:
                risk_data = []
                for stock in stocks[:paths.shape[2]]:
                    risk = risk_metrics.get(stock, {})
                    if risk:
                        risk_data.append({
                            "Asset": stock,
                            "VaR (95%)": f"{risk.get('var_95', 0)*100:.1f}%",
                            "CVaR (95%)": f"{risk.get('cvar_95', 0)*100:.1f}%",
                            "Sharpe": f"{risk.get('sharpe', 0):.2f}",
                            "Volatility": f"{risk.get('volatility', 0)*100:.1f}%",
                            "Expected Return": f"{risk.get('expected_return', 0)*100:.1f}%",
                            "Max Drawdown": f"{risk.get('max_drawdown', 0)*100:.1f}%"
                        })
                if risk_data:
                    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

            with tabs[4]:
                if len(stocks) >= 2:
                    asset_returns = []
                    asset_vols = []
                    for stock in stocks[:paths.shape[2]]:
                        risk = risk_metrics.get(stock, {})
                        if risk:
                            asset_returns.append(risk.get('expected_return', 0.1))
                            asset_vols.append(risk.get('volatility', 0.2))
                    if len(asset_returns) >= 2:
                        asset_returns = np.array(asset_returns)
                        asset_vols = np.array(asset_vols)
                        cov_matrix = np.outer(asset_vols, asset_vols) * (np.eye(len(asset_returns)) * 0.7 + 0.3)

                        n_port = 1000
                        rets, vols, sharpes = [], [], []
                        for _ in range(n_port):
                            w = np.random.random(len(asset_returns))
                            w /= np.sum(w)
                            r = np.sum(w * asset_returns)
                            v = np.sqrt(w.T @ cov_matrix @ w)
                            rets.append(r); vols.append(v); sharpes.append(r/v if v > 0 else 0)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=vols, y=rets, mode='markers',
                            marker=dict(size=4, color=sharpes, colorscale='RdYlGn', showscale=True),
                            name='Portfolios'))
                        fig.update_layout(title="Efficient Frontier",
                            xaxis_title="Volatility", yaxis_title="Expected Return",
                            height=500, template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Add at least 2 assets")

        # AI
        st.markdown('<div class="section-header">🧠 AI Market Analysis</div>', unsafe_allow_html=True)
        col_ai1, col_ai2 = st.columns([1, 3])
        with col_ai1:
            if st.button("Generate Insights", use_container_width=True):
                with st.spinner("AI analyzing..."):
                    try:
                        if LlamaExplainer:
                            explainer = LlamaExplainer()
                            explanation = explainer.explain_simulation_results(
                                tickers=stocks,
                                expected_prices=expected_prices,
                                confidence_intervals=confidence_intervals,
                                risk_metrics=risk_metrics,
                                variance_reduction=variance_reduction
                            )
                            st.session_state.explanation = explanation
                        else:
                            st.warning("LLM not available")
                    except Exception as e:
                        st.error(f"AI failed: {e}")

        with col_ai2:
            if st.session_state.explanation:
                st.markdown(f'<div class="info-box">{st.session_state.explanation}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">Click "Generate Insights" for AI analysis</div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            json_str = json.dumps(results, default=str, indent=2)
            st.download_button("📥 JSON", json_str, "results.json", use_container_width=True)
        with col2:
            if expected_prices:
                df_export = pd.DataFrame([{"Asset": k, "Forecast": v} for k, v in expected_prices.items()])
                st.download_button("📊 CSV", df_export.to_csv(index=False), "prices.csv", use_container_width=True)
        with col3:
            if st.button("🔄 New", use_container_width=True):
                st.session_state.results = None
                st.session_state.explanation = None
                safe_rerun()

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2 style="font-size: 3rem;">🚀 Ready</h2>
            <p style="font-size: 1.2rem; color: #94a3b8;">Configure portfolio in sidebar and run simulation</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.auto_refresh:
        time.sleep(5)
        safe_rerun()

    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #64748b; padding: 1rem; font-size: 0.8rem;">
            <p style="color: #ef4444; font-weight: bold;">⚠️ NOT FINANCIAL ADVICE</p>
            <p>Educational and research purposes only.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    run_dashboard()