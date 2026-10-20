"""
LLM explainer over Ollama's API (hosted ollama.com or a local server).

Settings (Streamlit secrets, then environment):
  OLLAMA_API_KEY   required for the hosted API
  OLLAMA_BASE_URL  default https://ollama.com/api   (local: http://localhost:11434/api)
  OLLAMA_MODEL     default gpt-oss:120b
If the model can't be reached, a clearly-labelled rule-based summary is returned instead.
"""

import hashlib
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from src.explainer.prompt_templates import PromptTemplates
from src.market_utils import is_number

load_dotenv(override=False)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://ollama.com/api"
DEFAULT_MODEL = "gpt-oss:120b"
SYSTEM_PROMPT = (
    "You are a careful financial analyst writing for an educational Monte Carlo dashboard. "
    "Use ONLY the numbers in the user's message; never invent prices, news or figures. "
    "If the data cannot answer something, say so. Write in Markdown. "
    "This is educational analysis, not financial advice."
)


def _setting(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        val = st.secrets.get(name, "")
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        pass
    return (os.environ.get(name, "") or default).strip()


class LlamaExplainer:
    MAX_CACHE = 64

    def __init__(self, model_name: Optional[str] = None, use_cache: bool = True,
                 base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 90):
        self.api_key = api_key if api_key is not None else _setting("OLLAMA_API_KEY")
        self.base_url = (base_url or _setting("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model_name = model_name or _setting("OLLAMA_MODEL", DEFAULT_MODEL)
        self.use_cache = use_cache
        self.timeout = timeout
        self.cache: "OrderedDict[str, str]" = OrderedDict()
        self.conversation_history: List[Dict] = []
        self.max_history = 10
        self.last_source: Optional[str] = None   # "llm" or "fallback"
        self.last_error: Optional[str] = None

        if self.is_hosted and not self.api_key:
            logger.warning("OLLAMA_API_KEY not set: explanations will use the rule-based fallback")

    # ------------------------------------------------------------
    @property
    def is_hosted(self) -> bool:
        return "ollama.com" in self.base_url

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _cache_key(self, prompt: str, temperature: float) -> str:
        return hashlib.sha256(f"{self.model_name}|{temperature}|{prompt}".encode()).hexdigest()

    def _call_api(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.4) -> Optional[str]:
        self.last_error = None
        if self.is_hosted and not self.api_key:
            self.last_error = "OLLAMA_API_KEY is not set"
            return None

        key = self._cache_key(prompt, temperature)
        if self.use_cache and key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            r = requests.post(f"{self.base_url}/chat", headers=self._headers(), json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout:
            self.last_error = f"timed out after {self.timeout}s"
            logger.warning(f"Ollama: {self.last_error}")
            return None
        except Exception as e:
            self.last_error = f"request failed: {e}"
            logger.warning(f"Ollama: {self.last_error}")
            return None

        if r.status_code != 200:
            self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            logger.warning(f"Ollama: {self.last_error}")
            return None
        try:
            data = r.json()
        except ValueError:
            self.last_error = "non-JSON response"
            return None

        text = ((data.get("message") or {}).get("content") or data.get("response") or "").strip()
        if not text:
            self.last_error = "empty response (a reasoning model may have used its whole token budget)"
            return None

        if self.use_cache:
            self.cache[key] = text
            while len(self.cache) > self.MAX_CACHE:
                self.cache.popitem(last=False)
        return text

    def check_connection(self) -> Tuple[bool, str]:
        """Cheap connectivity/model check (GET /tags)."""
        if self.is_hosted and not self.api_key:
            return False, "OLLAMA_API_KEY is not set"
        try:
            r = requests.get(f"{self.base_url}/tags", headers=self._headers(), timeout=15)
        except Exception as e:
            return False, f"request to {self.base_url} failed: {e}"
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} from {self.base_url}: {r.text[:200]}"
        try:
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except ValueError:
            names = []
        if names and not any(n == self.model_name or n.startswith(self.model_name) for n in names):
            return False, f"model '{self.model_name}' not listed. Available e.g.: {', '.join(names[:6])}"
        return True, f"Connected to {self.base_url}, model '{self.model_name}'"

    # ------------------------------------------------------------
    def explain_simulation_results(
        self,
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float = 0.0,
        user_question: Optional[str] = None,
        style: str = "comprehensive",
        horizon_days: Optional[int] = None,
    ) -> str:
        expected_prices = expected_prices or {}
        confidence_intervals = confidence_intervals or {}
        risk_metrics = risk_metrics or {}
        tickers = [t for t in (tickers or []) if t in risk_metrics or t in expected_prices]
        if not tickers:
            self.last_source = "fallback"
            self.last_error = "no simulation results"
            return "No simulation results to explain."

        insights = self._calculate_insights(tickers, expected_prices, risk_metrics)

        if user_question:
            prompt = PromptTemplates.answer_question(
                tickers, expected_prices, confidence_intervals, risk_metrics, variance_reduction,
                user_question, insights=insights, horizon_days=horizon_days)
        elif style == "concise":
            prompt = PromptTemplates.simulation_summary(
                tickers, expected_prices, confidence_intervals, risk_metrics, horizon_days=horizon_days)
        else:
            prompt = PromptTemplates.comprehensive_analysis(
                tickers, expected_prices, confidence_intervals, risk_metrics, insights, horizon_days)

        result = self._call_api(prompt)
        if result:
            self.last_source = "llm"
            self.conversation_history.extend([
                {"role": "user", "content": (user_question or "analysis")[:200]},
                {"role": "assistant", "content": result[:200]},
            ])
            self.conversation_history = self.conversation_history[-self.max_history:]
            return result

        self.last_source = "fallback"
        return self._get_enhanced_fallback(tickers, expected_prices, confidence_intervals, risk_metrics, insights)

    # ------------------------------------------------------------
    def _calculate_insights(self, tickers: List[str], expected_prices: Dict[str, float],
                            risk_metrics: Dict[str, Dict[str, float]]) -> Dict:
        """Mechanical screen. Daily VaR thresholds: < -3% High, < -1% Medium, else Low."""
        ins = {
            "best_performer": None, "worst_performer": None, "best_sharpe": None, "worst_sharpe": None,
            "average_sharpe": None, "average_var": None, "average_volatility": None,
            "total_assets": len(tickers), "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
            "risk_levels": {}, "signals": {}, "overall_sentiment": "Unknown",
        }
        sharpes, vars_, vols = {}, [], []
        for t in tickers:
            r = risk_metrics.get(t) or {}
            v = float(r["var_95"]) if is_number(r.get("var_95")) else None
            if v is not None:
                vars_.append(v)
                ins["risk_levels"][t] = "High" if v < -0.03 else ("Medium" if v < -0.01 else "Low")
            if is_number(r.get("volatility")):
                vols.append(float(r["volatility"]))
            if not is_number(r.get("sharpe")):
                continue
            s = float(r["sharpe"])
            sharpes[t] = s

            if s > 0.8:
                ins["bullish_count"] += 1
            elif s > 0.2:
                ins["neutral_count"] += 1
            else:
                ins["bearish_count"] += 1

            v_ok = v if v is not None else -1.0
            if s > 1.0 and v_ok > -0.02:
                ins["signals"][t] = "Strong"
            elif s > 0.5 and v_ok > -0.03:
                ins["signals"][t] = "Positive"
            elif s > 0.2:
                ins["signals"][t] = "Neutral"
            else:
                ins["signals"][t] = "Weak"

        if sharpes:
            best = max(sharpes, key=sharpes.get)
            worst = min(sharpes, key=sharpes.get)
            ins.update(best_performer=best, best_sharpe=sharpes[best],
                       worst_performer=worst, worst_sharpe=sharpes[worst],
                       average_sharpe=sum(sharpes.values()) / len(sharpes))
            if ins["bullish_count"] > ins["bearish_count"]:
                ins["overall_sentiment"] = "Bullish"
            elif ins["bearish_count"] > ins["bullish_count"]:
                ins["overall_sentiment"] = "Bearish"
            else:
                ins["overall_sentiment"] = "Neutral"
        if vars_:
            ins["average_var"] = sum(vars_) / len(vars_)
        if vols:
            ins["average_volatility"] = sum(vols) / len(vols)
        ins["recommendations"] = ins["signals"]  # backward-compatible alias
        return ins

    def _get_enhanced_fallback(self, tickers, expected_prices, confidence_intervals, risk_metrics, insights) -> str:
        P = PromptTemplates
        out = [f"_AI model unavailable ({self.last_error or 'no response'}). "
               "Rule-based summary computed from the simulation numbers._", "", "## 📊 Summary"]
        if insights.get("best_performer"):
            out.append(
                f"Screen tone: **{insights['overall_sentiment']}** ({insights['bullish_count']} strong-Sharpe, "
                f"{insights['neutral_count']} middling, {insights['bearish_count']} weak). "
                f"Highest Sharpe: **{insights['best_performer']}** ({P._fmt(insights['best_sharpe'])}); "
                f"lowest: **{insights['worst_performer']}** ({P._fmt(insights['worst_sharpe'])}).")
        else:
            out.append("Not enough risk metrics to summarise.")

        out += ["", "## 📈 Assets", ""]
        for t in tickers:
            r = risk_metrics.get(t) or {}
            out.append(
                f"**{t}**: expected {P._price(expected_prices.get(t), t)} "
                f"(95% range {P._ci(confidence_intervals.get(t), t)}); Sharpe {P._fmt(r.get('sharpe'))}, "
                f"daily VaR {P._pct(r.get('var_95'), '.2f')}, vol {P._pct(r.get('volatility'))}; "
                f"risk level {insights['risk_levels'].get(t, 'n/a')}, screen {insights['signals'].get(t, 'n/a')}.")
            out.append("")

        out += [
            "## ⚠️ Risks",
            f"- Average annualised volatility: {P._pct(insights.get('average_volatility'))}",
            f"- Average 1-day 95% VaR: {P._pct(insights.get('average_var'), '.2f')}",
            "- Drift is estimated from recent history and carried forward; it predicts future returns poorly.",
            "- Volatility and correlations are assumed stable; regime changes are not modelled.",
            "", "---", "*Not financial advice. Educational purposes only.*",
        ]
        return "\n".join(out)

    # ------------------------------------------------------------
    def clear_cache(self):
        self.cache.clear()

    def get_conversation_history(self) -> List[Dict]:
        return list(self.conversation_history)

    def clear_history(self):
        self.conversation_history = []