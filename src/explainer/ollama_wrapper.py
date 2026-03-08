"""
Ollama wrapper with API key support for DeepSeek models
"""

import requests
import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
import time

# Force load .env file from current directory
load_dotenv(override=True)

logger = logging.getLogger(__name__)


class LlamaExplainer:
    def __init__(self):
        # Get API key from environment variable
        self.api_key = os.environ.get("OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY", "")
        
        # Ollama API endpoint
        self.base_url = "https://api.ollama.com/api"
        self.model_name = "deepseek-v3.2"
        
        if not self.api_key:
            logger.warning("No OLLAMA_API_KEY found - AI features will use fallback summary")

    def explain_simulation_results(
        self,
        tickers,
        expected_prices,
        confidence_intervals,
        risk_metrics,
        variance_reduction,
        user_question=None
    ):
        if not self.api_key:
            return self._get_fallback_summary(tickers, expected_prices, risk_metrics)

        disclaimer = (
            "IMPORTANT DISCLAIMER: This is NOT financial advice. "
            "This is a probabilistic simulation for educational purposes only.\n\n"
        )

        prompt = user_question or self._build_prompt(
            tickers,
            expected_prices,
            confidence_intervals,
            risk_metrics,
            variance_reduction
        )

        # Try with shorter timeout first
        try:
            # Using generate endpoint with shorter timeout
            response = requests.post(
                f"{self.base_url}/generate",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "prompt": disclaimer + prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1000,  # Reduced for faster response
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=30  # Reduced timeout
            )

            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    result = data["response"]
                    result = result.strip()
                    
                    if "not financial advice" not in result.lower():
                        result = "⚠️ Not financial advice.\n\n" + result
                    
                    return result

        except requests.exceptions.Timeout:
            logger.warning("Ollama API timeout - using fallback summary")
            return self._get_fallback_summary(tickers, expected_prices, risk_metrics)
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return self._get_fallback_summary(tickers, expected_prices, risk_metrics)

    def _build_prompt(
        self,
        tickers,
        expected_prices,
        confidence_intervals,
        risk_metrics,
        variance_reduction
    ):
        # Shorter prompt for faster response
        prompt = "Analyze these Monte Carlo results briefly:\n\n"
        
        for ticker in tickers[:3]:  # Limit to 3 tickers for speed
            price = expected_prices.get(ticker, 0)
            ci_low, ci_high = confidence_intervals.get(ticker, (0, 0))
            risk = risk_metrics.get(ticker, {})
            
            prompt += (
                f"{ticker}: ${price:.2f} "
                f"[${ci_low:.2f}-${ci_high:.2f}] "
                f"VaR {risk.get('var_95', 0)*100:.1f}% "
                f"Sharpe {risk.get('sharpe', 0):.2f}\n"
            )
        
        prompt += f"\nVariance reduction: {variance_reduction*100:.1f}%\n"
        prompt += "Give 3 key insights:"
        
        return prompt

    def _get_fallback_summary(self, tickers, expected_prices, risk_metrics):
        """Fast fallback summary"""
        summary = "📊 **Quick Summary**\n\n"
        
        for ticker in tickers[:3]:
            if ticker in expected_prices:
                price = expected_prices[ticker]
                risk = risk_metrics.get(ticker, {})
                
                sharpe = risk.get('sharpe', 0)
                if sharpe > 1:
                    trend = "🟢"
                elif sharpe > 0:
                    trend = "🟡"
                else:
                    trend = "🔴"
                
                summary += (
                    f"{trend} **{ticker}**: ${price:.2f} | "
                    f"VaR: {risk.get('var_95', 0)*100:.1f}% | "
                    f"Sharpe: {sharpe:.2f}\n"
                )
        
        return summary