"""
Utility functions for the dashboard
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.market_utils import format_price, is_number

_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^&]{0,19}$")


def safe_float(value, default: float = float("nan")) -> float:
    return float(value) if is_number(value) else default


def parse_tickers(text: str, max_n: int = 10) -> Tuple[List[str], List[str], List[str]]:
    """Returns (tickers, invalid, dropped_over_limit). Deduplicates, keeps order."""
    tickers, invalid, seen = [], [], set()
    for tok in re.split(r"[,\s;]+", (text or "").upper()):
        if not tok:
            continue
        if not _TICKER_RE.match(tok):
            invalid.append(tok)
            continue
        if tok not in seen:
            seen.add(tok)
            tickers.append(tok)
    return tickers[:max_n], invalid, tickers[max_n:]


def format_currency(value, ticker: str = "", currency: Optional[str] = None) -> str:
    if currency is not None:
        return f"{currency}{float(value):,.2f}" if is_number(value) else "N/A"
    return format_price(value, ticker)


def format_percentage(value, decimals: int = 1, signed: bool = False) -> str:
    if not is_number(value):
        return "N/A"
    return f"{float(value) * 100:{'+' if signed else ''}.{decimals}f}%"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_metric_card(title: str, value: Any, delta: Any = None, help_text: str = None) -> Dict[str, Any]:
    card = {"title": title, "value": value}
    if delta is not None:
        card["delta"] = delta
    if help_text is not None:
        card["help"] = help_text
    return card


def prepare_asset_data(tickers: List[str], expected_prices: Dict[str, float],
                       current_prices: Dict[str, float],
                       risk_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Missing values are NaN (not 0, which looks like real data)."""
    rows = []
    for t in tickers:
        exp = safe_float(expected_prices.get(t))
        cur = safe_float(current_prices.get(t))
        risk = risk_metrics.get(t, {}) or {}
        rows.append({
            "Ticker": t,
            "Current": cur,
            "Expected": exp,
            "Change": exp / cur - 1 if np.isfinite(cur) and cur > 0 and np.isfinite(exp) else np.nan,
            "VaR (95%, daily)": safe_float(risk.get("var_95")),
            "Sharpe": safe_float(risk.get("sharpe")),
            "Volatility": safe_float(risk.get("volatility")),
        })
    return pd.DataFrame(rows)


def create_path_dataframe(paths, tickers: List[str], n_paths: int = 10) -> pd.DataFrame:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 3 or arr.size == 0:
        return pd.DataFrame(columns=["Path", "Day", "Ticker", "Price"])
    p, d, a = min(n_paths, arr.shape[0]), arr.shape[1], min(arr.shape[2], len(tickers))
    pi, di, ai = np.meshgrid(np.arange(p), np.arange(d), np.arange(a), indexing="ij")
    return pd.DataFrame({
        "Path": [f"Path {i + 1}" for i in pi.ravel()],
        "Day": di.ravel(),
        "Ticker": [tickers[j] for j in ai.ravel()],
        "Price": arr[:p, :, :a].ravel(),
    })