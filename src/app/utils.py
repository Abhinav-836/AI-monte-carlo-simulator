"""
Utility functions for the dashboard
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

def format_currency(value: float, currency: str = "$") -> str:
    """Format value as currency"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{currency}{value:,.2f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"

def create_metric_card(
    title: str,
    value: Any,
    delta: Any = None,
    help_text: str = None
) -> Dict[str, Any]:
    """Create a metric card for dashboard"""
    card = {
        "title": title,
        "value": value
    }
    if delta is not None:
        card["delta"] = delta
    if help_text is not None:
        card["help"] = help_text
    return card

def prepare_asset_data(
    tickers: List[str],
    expected_prices: Dict[str, float],
    current_prices: Dict[str, float],
    risk_metrics: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Prepare asset data for display"""
    data = []
    for ticker in tickers:
        expected = expected_prices.get(ticker, 0)
        current = current_prices.get(ticker, 0)
        risk = risk_metrics.get(ticker, {})
        
        data.append({
            "Ticker": ticker,
            "Current": current,
            "Expected": expected,
            "Change": (expected / current - 1) if current and current > 0 else 0,
            "VaR (95%)": risk.get("var_95", 0),
            "Sharpe": risk.get("sharpe", 0),
            "Volatility": risk.get("volatility", 0)
        })
    
    df = pd.DataFrame(data)
    return df

def create_path_dataframe(
    paths: List[List[List[float]]],
    tickers: List[str],
    n_paths: int = 10
) -> pd.DataFrame:
    """Create dataframe from path data for plotting"""
    paths_array = np.array(paths)
    n_assets = paths_array.shape[2]
    
    data = []
    for path_idx in range(min(n_paths, paths_array.shape[0])):
        for day in range(paths_array.shape[1]):
            for asset_idx in range(min(n_assets, len(tickers))):
                data.append({
                    "Path": f"Path {path_idx + 1}",
                    "Day": day,
                    "Ticker": tickers[asset_idx],
                    "Price": paths_array[path_idx, day, asset_idx]
                })
    
    return pd.DataFrame(data)