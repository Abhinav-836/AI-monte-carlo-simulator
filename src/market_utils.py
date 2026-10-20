"""
Small market helpers shared by the app and the explainer.
"""

import math
from typing import Optional

# yfinance-style suffix -> display currency. NOTE: LSE (.L) quotes are in pence (GBp).
_SUFFIX_CURRENCY = (
    (".NS", "₹"), (".BO", "₹"), (".L", "GBp "), (".HK", "HK$"), (".TO", "C$"),
    (".AX", "A$"), (".DE", "€"), (".PA", "€"), (".AS", "€"), (".T", "¥"),
)
_INDEX_CURRENCY = {"^NSEI": "₹", "^BSESN": "₹", "^NSEBANK": "₹", "^N225": "¥", "^FTSE": ""}


def currency_symbol(ticker: str) -> str:
    t = (ticker or "").upper()
    if t in _INDEX_CURRENCY:
        return _INDEX_CURRENCY[t]
    for suffix, symbol in _SUFFIX_CURRENCY:
        if t.endswith(suffix):
            return symbol
    return "$"


def is_number(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def format_price(value, ticker: str = "", decimals: Optional[int] = None) -> str:
    """Full-precision price in the ticker's currency, or an em dash if missing/invalid."""
    if not is_number(value) or float(value) <= 0:
        return "—"
    v = float(value)
    if decimals is None:
        decimals = 2 if v >= 1 else (6 if v < 0.01 else 4)
    return f"{currency_symbol(ticker)}{v:,.{decimals}f}"