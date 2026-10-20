# 🚀 AI-Powered Monte Carlo Stock Simulator

Monte Carlo simulation of global stock prices with optional generative models (WGAN-GP "DFMGAN"),
prediction-enhanced variance reduction (PEMC), portfolio optimisation, risk analytics and an
optional LLM that explains the results.

> ⚠️ **Disclaimer:** educational and research use only. Not financial advice.

---

## 📌 What it does

| Module | What it really does |
|---|---|
| **Monte Carlo** | Correlated GBM with jumps (Ledoit-Wolf covariance). Risk statistics use *all* simulated paths. |
| **DFMGAN** | WGAN-GP path generator + critic filter. **Optional**: used only when a trained checkpoint is loaded; otherwise the pipeline falls back to GBM. |
| **PEMC** | Neural control variate for option payoffs (Asian / barrier / variance swap / European). Standalone; see notebook 03. |
| **Risk & performance** | VaR/CVaR, Sharpe, Sortino, Calmar, drawdowns, capture ratios, Kelly, etc. |
| **Optimisation** | Max-Sharpe, min-vol, risk-parity, max-diversification, Black-Litterman, efficient frontier. |
| **LLM explainer** | Natural-language summary of simulation output (needs a local Ollama server). |

## 🌍 Market data

Prices come from a multi-tier fallback that never fabricates data:

- Current price: Twelve Data → Finnhub (US only) → yfinance → Yahoo chart API → Alpha Vantage → last real cached price
- Historical: Twelve Data → yfinance → Yahoo chart API → Alpha Vantage

If a ticker can't be fetched it is reported in `results["metadata"]["missing_tickers"]` instead of silently dropped.
Free-tier limits apply (Twelve Data: 8 requests/min). Non-US exchanges may not be covered by every free plan.
Yahoo can throttle shared cloud IPs.

## ⚙️ Setup

```bash
pip install -r requirements.txt          # add requirements-dev.txt for tests/notebooks
```

API keys (all optional, more keys = better coverage):

- **Local:** put them in `.env` (never committed)
- **Streamlit Cloud:** *Manage app → Settings → Secrets*

```toml
TWELVE_DATA_API_KEY = "..."
FINNHUB_API_KEY = "..."
ALPHA_VANTAGE_API_KEY = "..."
```

Run the app (adjust the filename to your entry point):

```bash
streamlit run app.py
```

## 🧪 Tests and notebooks

```bash
pytest -q                                              # no network, no API keys needed
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --output /tmp/nb01.ipynb
```

## 🧠 Training the GAN

Run `notebooks/02_dfmgan_training.ipynb`. It writes `data/models/dfmgan.pth`.
Commit that file (see `.gitignore`) so the deployed app can load it, then:

```python
pipeline = MonteCarloPipeline(n_assets=3, use_gan=True)
pipeline.load_models(checkpoint_path="data/models/dfmgan.pth")
```

The network sizes in the notebook must match the `DFMGANConfig` defaults, otherwise loading fails and the pipeline falls back to GBM.

## 🏗️ Architecture

```
 Market data (multi-provider fallback)
              │
              ▼
 Feature engineering (leak-free)
              │
     ┌────────┴─────────┐
     ▼                  ▼
 GBM + jumps      DFMGAN (optional, trained checkpoint)
     └────────┬─────────┘
              ▼
 Path simulation ──► Risk metrics / CIs ──► Optimiser
              │
              ▼
 LLM explainer (optional) ──► Streamlit UI
```

## ⚠️ Known limitations

- `variance_reduction` in the pipeline output compares simulated vs historical return variance. It is **not** PEMC variance reduction.
- Expected returns estimated from 1–2 years of history are very noisy; max-Sharpe weights are sensitive to them.
- Simulated paths are 60 steps (~59 trading days).

## 👨‍💻 Author

Abhinav Ashutosh