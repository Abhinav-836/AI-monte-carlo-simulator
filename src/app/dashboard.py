"""
Ultra-Modern Monte Carlo Dashboard - Premium Edition
Three.js Style + Glass Morphism + Advanced Analytics
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pipeline
try:
    from src.pipeline import MonteCarloPipeline
    print("✅ Import successful")
except ImportError as e:
    st.error(f"Failed to import MonteCarloPipeline: {e}")
    st.stop()

from src.explainer.ollama_wrapper import LlamaExplainer

# Page config
st.set_page_config(
    page_title="AI Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== OPTIMIZED 3JS STYLE CSS =====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Main header with glass effect */
    .main-header {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        padding: 2rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
    }
    
    /* Glass panel effect */
    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #6366f1 0%, transparent 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        color: white;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
    }
    
    /* Welcome cards */
    .welcome-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        transition: transform 0.3s;
    }
    
    .welcome-card:hover {
        transform: translateY(-5px);
        border-color: #6366f1;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 4px;
    }
    
    /* Risk disclaimer - only at bottom now */
    .risk-disclaimer {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #fecaca;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

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
    st.session_state.stocks_input = "AAPL, MSFT, GOOGL, NVDA, META, RELIANCE.NS, TCS.NS"

# ===================== HELPER FUNCTIONS =====================
def safe_rerun():
    """Safely rerun the app"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except:
            st.warning("Please refresh the page")

def format_currency(value):
    """Format currency values"""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    return f"${value:.2f}"

# ===================== HEADER =====================
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-size: 2.5rem;">🚀 AI Monte Carlo</h1>
            <p style="margin:0.5rem 0 0 0; opacity:0.9;">Deep Learning Powered Portfolio Simulation</p>
        </div>
        <div style="display: flex; gap: 0.8rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem;">🌍 Global</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem;">🤖 AI</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.9rem;">📊 Options</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Show error if any
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    if st.button("Clear Error"):
        st.session_state.error_message = None
        safe_rerun()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    # Quick stock selection
    st.markdown("**Quick Select**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇺🇸 US Tech", use_container_width=True):
            st.session_state.stocks_input = "AAPL, MSFT, GOOGL, NVDA, META"
    with col2:
        if st.button("🇮🇳 India NSE", use_container_width=True):
            st.session_state.stocks_input = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS"
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇬🇧 UK", use_container_width=True):
            st.session_state.stocks_input = "BP.L, HSBA.L, GSK.L, AZN.L, DGE.L"
    with col2:
        if st.button("₿ Crypto", use_container_width=True):
            st.session_state.stocks_input = "BTC-USD, ETH-USD, BNB-USD, SOL-USD"
    
    # Stock input
    stocks_input = st.text_area(
        "**Enter Symbols**",
        value=st.session_state.stocks_input,
        height=100,
        help="Format: AAPL, MSFT.NS, 0700.HK, BTC-USD"
    )
    
    # Parse stocks
    if "," in stocks_input:
        stocks = [s.strip().upper() for s in stocks_input.split(",") if s.strip()]
    else:
        stocks = [s.strip().upper() for s in stocks_input.split("\n") if s.strip()]
    
    st.session_state.stocks_input = ", ".join(stocks)
    
    st.markdown("---")
    
    # Simulation Settings
    st.markdown("**Simulation**")
    
    n_sims = st.slider(
        "Number of Paths",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100
    )
    
    use_gan = st.checkbox(
        "🤖 AI Generator",
        value=True,
        help="Use Deep Learning for realistic paths"
    )
    
    filter_pct = st.slider(
        "Keep Top %",
        min_value=1,
        max_value=50,
        value=10
    ) / 100
    
    st.markdown("**Data**")
    
    period = st.selectbox(
        "Time Period",
        options=["6mo", "1y", "2y", "5y"],
        index=2
    )
    
    include_options = st.checkbox(
        "📋 Show Options",
        value=True
    )
    
    # Advanced settings
    with st.expander("Advanced"):
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01
        )
    
    # Run button
    run_button = st.button(
        "🚀 RUN SIMULATION",
        use_container_width=True,
        type="primary"
    )
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.session_state.pipeline = None
        safe_rerun()
    
    st.caption(f"Tracking: {len(stocks)} assets")

# ===================== INITIALIZE PIPELINE =====================
if st.session_state.pipeline is None and stocks:
    with st.spinner("Initializing AI Engine..."):
        try:
            pipeline = MonteCarloPipeline(
                n_assets=len(stocks),
                n_simulations=n_sims,
                filter_top_k=filter_pct,
                use_gan=use_gan
            )
            pipeline.load_models()
            st.session_state.pipeline = pipeline
            st.session_state.error_message = None
            st.success("✅ AI Engine Ready!")
            time.sleep(1)
            safe_rerun()
        except Exception as e:
            st.session_state.error_message = str(e)
            st.error(f"Initialization failed: {e}")

# ===================== RUN SIMULATION =====================
if run_button and st.session_state.pipeline and stocks:
    with st.spinner("AI simulating market scenarios..."):
        try:
            pipeline = st.session_state.pipeline
            pipeline.n_simulations = n_sims
            pipeline.filter_top_k = filter_pct
            pipeline.use_gan = use_gan
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run simulation
            results = pipeline.run_simulation(
                tickers=stocks,
                period=period,
                use_real_options=include_options
            )
            
            # Update progress
            messages = [
                "Analyzing patterns...",
                "Generating paths...",
                "Monte Carlo...",
                "Optimizing...",
                "Calculating risk..."
            ]
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i % 20 == 0:
                    status_text.text(random.choice(messages))
                time.sleep(0.01)
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.results = results
            st.session_state.explanation = None
            st.session_state.error_message = None
            
            st.success("✅ Simulation Complete!")
            
        except Exception as e:
            st.session_state.error_message = str(e)
            st.error(f"Simulation failed: {e}")

# ===================== DISPLAY RESULTS =====================
if st.session_state.results:
    results = st.session_state.results
    metadata = results.get("metadata", {})
    expected_prices = results.get("expected_prices", {})
    current_prices = results.get("current_prices", {})
    confidence_intervals = results.get("confidence_intervals", {})
    risk_metrics = results.get("risk_metrics", {})
    option_prices = results.get("option_prices", {})
    
    # Metrics Row
    st.markdown("### 📊 Live Stats")
    
    cols = st.columns(6)
    metrics_data = [
        ("Paths", f"{metadata.get('n_simulations', 0):,}", "🔄"),
        ("Filtered", f"{metadata.get('filtered_paths', 0):,}", "🎯"),
        ("Time", f"{metadata.get('computation_time', 0):.1f}s", "⏱️"),
        ("Variance ↓", f"{results.get('variance_reduction', 0)*100:.1f}%", "📉"),
        ("Data", "Live", "📡"),
        ("AI Mode", "✅" if metadata.get('use_gan', False) else "⚡", "🤖")
    ]
    
    for col, (label, value, emoji) in zip(cols, metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">{emoji}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Price Forecast Table
    st.markdown('<div class="section-header">💰 Price Forecast</div>', unsafe_allow_html=True)
    
    price_data = []
    for stock in stocks:
        if stock in expected_prices:
            current = current_prices.get(stock, 0)
            expected = expected_prices[stock]
            change = ((expected - current) / current * 100) if current else 0
            ci = confidence_intervals.get(stock, [0, 0])
            
            # Trend emoji
            trend = "🚀" if change > 5 else "📈" if change > 0 else "📉" if change > -5 else "💥"
            
            price_data.append({
                "Asset": stock,
                "Current": format_currency(current),
                "Forecast": format_currency(expected),
                "Trend": f"{trend} {change:+.1f}%",
                "Range": format_currency(ci[1] - ci[0])
            })
    
    if price_data:
        df_prices = pd.DataFrame(price_data)
        st.dataframe(df_prices, use_container_width=True, hide_index=True)
    
    # Charts
    st.markdown('<div class="section-header">📈 Analysis</div>', unsafe_allow_html=True)
    
    if "path_sample" in results and results["path_sample"]:
        paths = np.array(results["path_sample"])
        
        tab1, tab2, tab3 = st.tabs(["📊 Paths", "📉 Distribution", "📋 Options"])
        
        with tab1:
            selected = st.multiselect(
                "Select assets",
                options=stocks[:paths.shape[2]],
                default=stocks[:min(3, paths.shape[2])]
            )
            
            if selected:
                fig = go.Figure()
                colors = ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']
                
                for i, stock in enumerate(selected):
                    idx = stocks.index(stock)
                    mean_path = np.mean(paths[:, :, idx], axis=0)
                    upper = np.percentile(paths[:, :, idx], 95, axis=0)
                    lower = np.percentile(paths[:, :, idx], 5, axis=0)
                    color = colors[i % len(colors)]
                    
                    # Confidence band
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))), y=upper,
                        line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))), y=lower,
                        fill='tonexty', fillcolor=f'rgba{tuple(int(color.lstrip("#")[j:j+2], 16) for j in (0,2,4)) + (0.1,)}',
                        line=dict(width=0), showlegend=False, hoverinfo='skip'
                    ))
                    
                    # Mean line
                    fig.add_trace(go.Scatter(
                        x=list(range(len(mean_path))), y=mean_path,
                        name=stock, line=dict(color=color, width=3)
                    ))
                
                fig.update_layout(
                    title="Price Paths with 95% Confidence",
                    xaxis_title="Days", yaxis_title="Price ($)",
                    hovermode='x unified', height=500,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            stock = st.selectbox("Select asset", options=stocks[:paths.shape[2]])
            if stock:
                idx = stocks.index(stock)
                final_prices = paths[:, -1, idx]
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "Box Plot"))
                
                fig.add_trace(go.Histogram(x=final_prices, nbinsx=50, marker_color='#6366f1'), row=1, col=1)
                fig.add_trace(go.Box(y=final_prices, name=stock, marker_color='#8b5cf6'), row=1, col=2)
                
                fig.update_layout(height=400, template='plotly_dark', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if include_options and option_prices:
                for stock, opt_data in option_prices.items():
                    if opt_data and opt_data.get('calls'):
                        with st.expander(f"{stock} Options"):
                            st.json(opt_data)
            else:
                st.info("No options data available")
    
    # AI Analysis
    st.markdown('<div class="section-header">🧠 AI Analysis</div>', unsafe_allow_html=True)
    
    if st.button("Generate Insights"):
        with st.spinner("AI analyzing..."):
            try:
                explainer = LlamaExplainer()
                explanation = explainer.explain_simulation_results(
                    tickers=stocks,
                    expected_prices=expected_prices,
                    confidence_intervals=confidence_intervals,
                    risk_metrics=risk_metrics,
                    variance_reduction=results.get("variance_reduction", 0)
                )
                st.session_state.explanation = explanation
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
    
    if st.session_state.explanation:
        st.markdown(f'<div class="info-box">{st.session_state.explanation}</div>', unsafe_allow_html=True)
    
    # Export
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_str = json.dumps(results, default=str, indent=2)
        st.download_button("📥 Export JSON", json_str, "results.json", use_container_width=True)
    
    with col2:
        if expected_prices:
            df_export = pd.DataFrame([{"Asset": k, "Forecast": v} for k, v in expected_prices.items()])
            st.download_button("📊 Export CSV", df_export.to_csv(index=False), "prices.csv", use_container_width=True)
    
    with col3:
        if st.button("🔄 New", use_container_width=True):
            safe_rerun()

else:
    # Welcome Screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2 style="font-size: 3rem; margin-bottom: 1rem;">🚀 Ready</h2>
        <p style="font-size: 1.2rem; margin-bottom: 3rem; color: #94a3b8;">
            Configure in sidebar
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div class="welcome-card">
                <div class="feature-icon">📊</div>
                <h3>10K+ Paths</h3>
                <p>Monte Carlo</p>
            </div>
            <div class="welcome-card">
                <div class="feature-icon">🤖</div>
                <h3>AI Generator</h3>
                <p>Deep Learning</p>
            </div>
            <div class="welcome-card">
                <div class="feature-icon">🌍</div>
                <h3>Global</h3>
                <p>50+ Markets</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer with strong disclaimer (only here at the bottom)
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; padding: 1rem; font-size: 0.8rem;">
        <p style="color: #ef4444; font-weight: bold;">⚠️ NOT FINANCIAL ADVICE - USE AT YOUR OWN RISK</p>
        <p>This tool is for educational and research purposes only. Results are simulated and based on mathematical models.</p>
        <p>Past performance does not indicate future results. Always consult a qualified financial advisor.</p>
        <p style="margin-top: 1rem;">© 2026 AI Monte Carlo Simulator - Research Version</p>
    </div>
    """,
    unsafe_allow_html=True
)