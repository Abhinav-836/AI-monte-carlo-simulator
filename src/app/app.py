"""
Main Application Entry Point
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path - FIXED for proper import
ROOT_DIR = Path(__file__).parent.parent.parent  # This goes up to MonteCarlo-Hybrid-Explainer
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now import from src
from src.app.dashboard import run_dashboard


def main():
    """Run the application"""
    run_dashboard()


if __name__ == "__main__":
    main()