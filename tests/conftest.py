"""
Pytest configuration: import path, Streamlit stubs, no network, no real API keys, fixed seeds.
Place in tests/ (ROOT_DIR is its parent).
"""

import os
import sys
from contextlib import nullcontext

import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class MockSessionState(dict):
    """dict with attribute access, like the real st.session_state
    (missing attribute -> AttributeError; `in` and `.get` work, unlike the old mock which looped forever)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"st.session_state has no attribute {name!r}") from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None


@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    import streamlit as st

    monkeypatch.setattr(st, "session_state", MockSessionState(), raising=False)
    monkeypatch.setattr(st, "secrets", {}, raising=False)  # never read a local secrets.toml in tests

    noop = lambda *a, **k: None
    for name in ("markdown", "title", "header", "subheader", "write", "caption", "warning",
                 "error", "info", "success", "metric", "dataframe", "plotly_chart", "pyplot"):
        monkeypatch.setattr(st, name, noop, raising=False)
    monkeypatch.setattr(st, "button", lambda *a, **k: False, raising=False)
    monkeypatch.setattr(st, "text_input", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(st, "slider", lambda *a, **k: 1000, raising=False)
    monkeypatch.setattr(st, "spinner", lambda *a, **k: nullcontext(), raising=False)  # usable in `with`


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    for key in ("TWELVE_DATA_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any un-mocked `requests` call fails loudly instead of hitting a real provider."""
    import requests

    def _blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", _blocked)


@pytest.fixture(autouse=True)
def fixed_seeds():
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass