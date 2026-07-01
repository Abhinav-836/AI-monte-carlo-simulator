"""
Live Dashboard Component - Real-Time Price & Performance Tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional


class LiveDashboard:
    """
    Real-time dashboard component for streaming updates
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.is_running = False
        self.update_thread = None
        self.last_update = None
        
    def start(self):
        """Start the live dashboard"""
        self.is_running = True
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
    
    def stop(self):
        """Stop the live dashboard"""
        self.is_running = False
        
    def _update_loop(self):
        """Background update loop"""
        while self.is_running:
            self.last_update = datetime.now()
            time.sleep(1)  # Update every second
    
    def render_price_ticker(self, tickers: List[str]):
        """Render live price ticker"""
        prices = self.pipeline.get_live_prices()
        
        if not prices:
            st.info("Waiting for live data...")
            return
        
        cols = st.columns(min(len(tickers), 6))
        for i, ticker in enumerate(tickers[:6]):
            if ticker in prices:
                price = prices[ticker]
                history = self.pipeline.get_price_history(ticker)
                
                # Calculate change
                change = 0
                if len(history) >= 2:
                    change = ((history[-1]['price'] - history[-2]['price']) / history[-2]['price']) * 100
                
                color = "#10b981" if change >= 0 else "#ef4444"
                arrow = "↑" if change >= 0 else "↓"
                
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                        <div style="font-size: 0.8rem; color: #94a3b8;">{ticker}</div>
                        <div style="font-size: 1.5rem; font-weight: 600;">${price:.2f}</div>
                        <div style="font-size: 1rem; color: {color};">{arrow} {change:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_price_chart(self, ticker: str, height: int = 400):
        """Render live price chart for a ticker"""
        history = self.pipeline.get_price_history(ticker)
        
        if not history:
            st.info(f"No history data for {ticker}")
            return
        
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['price'],
            mode='lines',
            name=ticker,
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        
        # Moving average
        if len(df) > 20:
            ma = df['price'].rolling(20).mean()
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=ma,
                mode='lines',
                name='MA(20)',
                line=dict(color='#8b5cf6', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{ticker} Live Price",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=height,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render performance metrics"""
        metrics = self.pipeline.get_performance_metrics()
        
        if not metrics:
            return
        
        cols = st.columns(4)
        metrics_data = [
            ("Total Runs", f"{metrics['total_runs']}"),
            ("Avg Time", f"{metrics['avg_computation_time']:.1f}s"),
            ("Variance Reduction", f"{metrics['avg_variance_reduction']*100:.1f}%")
        ]
        
        for col, (label, value) in zip(cols, metrics_data):
            with col:
                st.metric(label, value)
    
    def render_heatmap(self, tickers: List[str], risk_metrics: Dict):
        """Render risk heatmap"""
        if not risk_metrics:
            return
        
        data = []
        for ticker in tickers:
            if ticker in risk_metrics:
                risk = risk_metrics[ticker]
                data.append({
                    'Asset': ticker,
                    'VaR': risk.get('var_95', 0) * 100,
                    'Volatility': risk.get('volatility', 0) * 100,
                    'Sharpe': risk.get('sharpe', 0),
                    'Drawdown': risk.get('max_drawdown', 0) * 100
                })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Risk-Return", "VaR & Drawdown")
        )
        
        # Risk-Return scatter
        fig.add_trace(go.Scatter(
            x=df['Volatility'],
            y=df['Sharpe'],
            mode='markers+text',
            text=df['Asset'],
            textposition='top center',
            marker=dict(
                size=df['VaR'] * 2,
                color=df['Sharpe'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Risk-Return'
        ), row=1, col=1)
        
        # VaR & Drawdown bar chart
        fig.add_trace(go.Bar(
            x=df['Asset'],
            y=df['VaR'],
            name='VaR',
            marker_color='#ef4444'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=df['Asset'],
            y=df['Drawdown'],
            name='Drawdown',
            marker_color='#f59e0b'
        ), row=1, col=2)
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations(self, risk_metrics: Dict):
        """Render AI-powered recommendations"""
        if not risk_metrics:
            return
        
        recommendations = []
        
        for ticker, metrics in risk_metrics.items():
            sharpe = metrics.get('sharpe', 0)
            var = metrics.get('var_95', 0)
            drawdown = metrics.get('max_drawdown', 0)
            
            # Generate recommendation
            if sharpe > 1.5 and var > -0.02:
                signal = "🟢 STRONG BUY"
                details = "Excellent risk-adjusted returns"
            elif sharpe > 0.8 and var > -0.03:
                signal = "🟡 BUY"
                details = "Good risk-return profile"
            elif sharpe > 0.2 and var > -0.04:
                signal = "🟠 HOLD"
                details = "Moderate performance"
            else:
                signal = "🔴 SELL/REDUCE"
                details = "Poor risk-adjusted returns"
            
            if drawdown < -0.3:
                signal = "🔴 SELL/REDUCE"
                details = "High drawdown risk"
            
            recommendations.append({
                'Asset': ticker,
                'Signal': signal,
                'Details': details,
                'Sharpe': f"{sharpe:.2f}",
                'VaR': f"{var*100:.1f}%"
            })
        
        df_rec = pd.DataFrame(recommendations)
        st.dataframe(df_rec, use_container_width=True, hide_index=True)