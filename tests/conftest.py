"""
Pytest configuration file - fixes import paths and prevents Streamlit issues
"""

import os
import sys
import pytest

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Mock Streamlit for tests
@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """Mock Streamlit session state to prevent errors in tests"""
    import streamlit as st
    
    class MockSessionState:
        def __init__(self):
            self._dict = {}
        
        def __getitem__(self, key):
            return self._dict.get(key)
        
        def __getattr__(self, name):
            return self._dict.get(name)
        
        def __setitem__(self, key, value):
            self._dict[key] = value
        
        def __setattr__(self, name, value):
            if name == '_dict':
                super().__setattr__(name, value)
            else:
                self._dict[name] = value
    
    # Mock session_state
    monkeypatch.setattr(st, "session_state", MockSessionState(), raising=False)
    
    # Mock other streamlit functions that might be called
    monkeypatch.setattr(st, "markdown", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(st, "title", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(st, "metric", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(st, "dataframe", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(st, "button", lambda *args, **kwargs: False, raising=False)
    monkeypatch.setattr(st, "text_input", lambda *args, **kwargs: "", raising=False)
    monkeypatch.setattr(st, "slider", lambda *args, **kwargs: 1000, raising=False)
    monkeypatch.setattr(st, "spinner", lambda *args, **kwargs: None, raising=False)