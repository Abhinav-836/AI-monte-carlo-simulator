"""
Advanced Ollama Wrapper with Enhanced Insights & Better Prompting
"""

import requests
import logging
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import time
import json
from functools import lru_cache
import hashlib
import numpy as np

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class LlamaExplainer:
    """
    Advanced LLM wrapper with enhanced insights and better prompting
    """
    
    def __init__(self, model_name: str = "deepseek-v3.2", use_cache: bool = True):
        self.api_key = os.environ.get("OLLAMA_API_KEY", "")
        self.base_url = "https://api.ollama.com/api"
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache = {}
        self.conversation_history = []
        self.max_history = 10
        
        if not self.api_key:
            logger.warning("No OLLAMA_API_KEY found - using enhanced fallback")
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _call_api(
        self,
        prompt: str,
        max_tokens: int = 800,
        temperature: float = 0.7,
        timeout: int = 30
    ) -> Optional[str]:
        """Call Ollama API with caching"""
        
        if not self.api_key:
            return None
        
        cache_key = self._get_cache_key(prompt)
        if self.use_cache and cache_key in self.cache:
            logger.info("📦 Using cached response")
            return self.cache[cache_key]
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_k": 40,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", "")
                
                if self.use_cache:
                    self.cache[cache_key] = result
                
                return result
            
        except requests.exceptions.Timeout:
            logger.warning("Ollama API timeout")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
        
        return None
    
    def explain_simulation_results(
        self,
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float,
        user_question: Optional[str] = None,
        style: str = "comprehensive"
    ) -> str:
        """Enhanced explanation with comprehensive analysis"""
        
        # Calculate additional insights
        insights = self._calculate_insights(tickers, expected_prices, risk_metrics)
        
        if user_question:
            prompt = self._build_qa_prompt(
                tickers, expected_prices, confidence_intervals,
                risk_metrics, variance_reduction, user_question, insights
            )
        else:
            prompt = self._build_comprehensive_prompt(
                tickers, expected_prices, confidence_intervals,
                risk_metrics, variance_reduction, insights
            )
        
        # Add disclaimer
        disclaimer = "⚠️ IMPORTANT: This is NOT financial advice. For educational purposes only.\n\n"
        
        full_prompt = disclaimer + prompt
        
        # Try API
        result = self._call_api(full_prompt, max_tokens=1000, temperature=0.7)
        
        if result:
            # Store in conversation history
            self.conversation_history.append({
                'role': 'user' if user_question else 'system',
                'content': prompt[:200] + '...'
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': result[:200] + '...'
            })
            
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            return result
        
        # Enhanced fallback
        return self._get_enhanced_fallback(tickers, expected_prices, risk_metrics, insights)
    
    def _calculate_insights(
        self,
        tickers: List[str],
        expected_prices: Dict[str, float],
        risk_metrics: Dict[str, Dict[str, float]]
    ) -> Dict:
        """Calculate additional insights from the data"""
        
        insights = {
            'best_performer': None,
            'worst_performer': None,
            'best_sharpe': -float('inf'),
            'worst_sharpe': float('inf'),
            'average_sharpe': 0,
            'average_var': 0,
            'total_assets': len(tickers),
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'risk_levels': {},
            'recommendations': {}
        }
        
        sharpe_sum = 0
        var_sum = 0
        valid_count = 0
        
        for ticker in tickers:
            if ticker in risk_metrics:
                risk = risk_metrics[ticker]
                sharpe = risk.get('sharpe', 0)
                var = risk.get('var_95', 0)
                
                sharpe_sum += sharpe
                var_sum += var
                valid_count += 1
                
                # Best performer
                if sharpe > insights['best_sharpe']:
                    insights['best_sharpe'] = sharpe
                    insights['best_performer'] = ticker
                
                # Worst performer
                if sharpe < insights['worst_sharpe']:
                    insights['worst_sharpe'] = sharpe
                    insights['worst_performer'] = ticker
                
                # Sentiment
                if sharpe > 0.8:
                    insights['bullish_count'] += 1
                elif sharpe > 0.2:
                    insights['neutral_count'] += 1
                else:
                    insights['bearish_count'] += 1
                
                # Risk level
                if var < -0.03:
                    insights['risk_levels'][ticker] = 'High'
                elif var < -0.01:
                    insights['risk_levels'][ticker] = 'Medium'
                else:
                    insights['risk_levels'][ticker] = 'Low'
                
                # Recommendations
                if sharpe > 1.0 and var > -0.02:
                    insights['recommendations'][ticker] = 'Strong Buy'
                elif sharpe > 0.5 and var > -0.03:
                    insights['recommendations'][ticker] = 'Buy'
                elif sharpe > 0.2:
                    insights['recommendations'][ticker] = 'Hold'
                else:
                    insights['recommendations'][ticker] = 'Sell/Reduce'
        
        if valid_count > 0:
            insights['average_sharpe'] = sharpe_sum / valid_count
            insights['average_var'] = var_sum / valid_count
        
        # Overall market sentiment
        if insights['bullish_count'] > insights['bearish_count']:
            insights['overall_sentiment'] = 'Bullish'
        elif insights['bearish_count'] > insights['bullish_count']:
            insights['overall_sentiment'] = 'Bearish'
        else:
            insights['overall_sentiment'] = 'Neutral'
        
        return insights
    
    def _build_comprehensive_prompt(
        self,
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float,
        insights: Dict
    ) -> str:
        """Build comprehensive analysis prompt"""
        
        # Format asset data with emojis and clear formatting
        asset_section = []
        for ticker in tickers[:5]:
            price = expected_prices.get(ticker, 0)
            ci = confidence_intervals.get(ticker, [0, 0])
            risk = risk_metrics.get(ticker, {})
            
            recommendation = insights['recommendations'].get(ticker, 'Hold')
            risk_level = insights['risk_levels'].get(ticker, 'Medium')
            
            # Emoji for recommendation
            rec_emoji = {
                'Strong Buy': '🟢',
                'Buy': '📈',
                'Hold': '🟡',
                'Sell/Reduce': '🔴'
            }.get(recommendation, '⚪')
            
            asset_section.append(
                f"{rec_emoji} **{ticker}**\n"
                f"  • Price: ${price:.2f} (Range: ${ci[0]:.2f}-${ci[1]:.2f})\n"
                f"  • Sharpe: {risk.get('sharpe', 0):.2f} | VaR: {risk.get('var_95', 0)*100:.1f}%\n"
                f"  • Volatility: {risk.get('volatility', 0)*100:.1f}% | Return: {risk.get('expected_return', 0)*100:.1f}%\n"
                f"  • Risk Level: {risk_level} | Recommendation: {recommendation}\n"
            )
        
        asset_text = "\n".join(asset_section)
        
        # Insights summary
        insights_section = f"""
MARKET INSIGHTS:
- Overall Sentiment: {insights['overall_sentiment']}
- Best Performer: {insights['best_performer']} (Sharpe: {insights['best_sharpe']:.2f})
- Worst Performer: {insights['worst_performer']} (Sharpe: {insights['worst_sharpe']:.2f})
- Average Sharpe: {insights['average_sharpe']:.2f}
- Bullish Assets: {insights['bullish_count']} | Bearish: {insights['bearish_count']} | Neutral: {insights['neutral_count']}
- Model Performance: {variance_reduction*100:.1f}% variance reduction
"""

        return f"""
Act as a senior financial analyst. Provide a comprehensive market analysis.

ASSET DATA:
{asset_text}

{insights_section}

Please provide analysis in the following format:

## 📊 EXECUTIVE SUMMARY
(2-3 sentences summarizing the overall market picture)

## 📈 INDIVIDUAL ASSET ANALYSIS
(For each asset, provide 2-3 lines of analysis covering performance, risk, and recommendation)

## ⚠️ KEY RISKS
(List 3-4 key risks to watch)

## 💡 STRATEGIC INSIGHTS
(Provide 2-3 actionable insights)

## 🎯 RECOMMENDATIONS
(Summary table of recommendations)

Keep it professional, concise, and actionable. Use proper financial terminology.
"""
    
    def _build_qa_prompt(
        self,
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float,
        question: str,
        insights: Dict
    ) -> str:
        """Build Q&A prompt"""
        
        context = []
        for ticker in tickers[:3]:
            price = expected_prices.get(ticker, 0)
            ci = confidence_intervals.get(ticker, [0, 0])
            risk = risk_metrics.get(ticker, {})
            
            context.append(
                f"{ticker}: ${price:.2f} (Range: ${ci[0]:.2f}-${ci[1]:.2f}), "
                f"Sharpe: {risk.get('sharpe', 0):.2f}, "
                f"VaR: {risk.get('var_95', 0)*100:.1f}%, "
                f"Recommendation: {insights['recommendations'].get(ticker, 'Hold')}"
            )
        
        return f"""
CONTEXT:
{chr(10).join(context)}

Overall Sentiment: {insights['overall_sentiment']}
Best Performer: {insights['best_performer']} (Sharpe: {insights['best_sharpe']:.2f})
Worst Performer: {insights['worst_performer']} (Sharpe: {insights['worst_sharpe']:.2f})
Variance Reduction: {variance_reduction*100:.1f}%

QUESTION: {question}

Provide a clear, detailed answer based on the data.
Be specific and actionable.
"""
    
    def _get_enhanced_fallback(
        self,
        tickers: List[str],
        expected_prices: Dict[str, float],
        risk_metrics: Dict[str, Dict[str, float]],
        insights: Dict
    ) -> str:
        """Enhanced fallback with detailed analysis"""
        
        summary = "📊 **Comprehensive Market Analysis**\n\n"
        
        # Executive Summary
        summary += f"## 📊 EXECUTIVE SUMMARY\n"
        summary += f"Market sentiment is **{insights['overall_sentiment']}** with {insights['bullish_count']} bullish and {insights['bearish_count']} bearish assets. "
        summary += f"The best performer is **{insights['best_performer']}** with Sharpe {insights['best_sharpe']:.2f}, "
        summary += f"while **{insights['worst_performer']}** is the weakest with Sharpe {insights['worst_sharpe']:.2f}.\n\n"
        
        # Individual Asset Analysis
        summary += f"## 📈 INDIVIDUAL ASSET ANALYSIS\n\n"
        
        for ticker in tickers[:3]:
            if ticker in expected_prices and ticker in risk_metrics:
                price = expected_prices[ticker]
                risk = risk_metrics[ticker]
                rec = insights['recommendations'].get(ticker, 'Hold')
                risk_level = insights['risk_levels'].get(ticker, 'Medium')
                
                summary += f"**{ticker}**\n"
                summary += f"• Expected Price: ${price:.2f}\n"
                summary += f"• Sharpe: {risk.get('sharpe', 0):.2f} | VaR: {risk.get('var_95', 0)*100:.1f}%\n"
                summary += f"• Volatility: {risk.get('volatility', 0)*100:.1f}% | Return: {risk.get('expected_return', 0)*100:.1f}%\n"
                summary += f"• Risk Level: {risk_level} | Recommendation: {rec}\n\n"
        
        # Key Risks
        summary += f"## ⚠️ KEY RISKS\n"
        summary += f"1. **Market Volatility**: Average volatility is {insights['average_sharpe']:.2f} Sharpe ratio\n"
        summary += f"2. **Asset Concentration**: {insights['total_assets']} assets in portfolio\n"
        summary += f"3. **Downside Risk**: Average VaR is {insights['average_var']*100:.1f}%\n"
        summary += f"4. **Performance Dispersion**: Wide range between best and worst performers\n\n"
        
        # Strategic Insights
        summary += f"## 💡 STRATEGIC INSIGHTS\n"
        if insights['overall_sentiment'] == 'Bullish':
            summary += "1. Consider increasing exposure to top performers\n"
            summary += "2. Look for pullbacks in strong assets\n"
            summary += "3. Maintain stop-losses on weaker positions\n"
        elif insights['overall_sentiment'] == 'Bearish':
            summary += "1. Consider reducing overall exposure\n"
            summary += "2. Focus on defensive assets\n"
            summary += "3. Consider hedging strategies\n"
        else:
            summary += "1. Maintain balanced allocation\n"
            summary += "2. Watch for breakout signals\n"
            summary += "3. Keep tight risk controls\n"
        
        # Recommendations Table
        summary += f"\n## 🎯 RECOMMENDATIONS\n"
        summary += "| Asset | Recommendation | Risk Level | Sharpe |\n"
        summary += "|-------|---------------|------------|--------|\n"
        
        for ticker in tickers[:3]:
            if ticker in risk_metrics:
                risk = risk_metrics[ticker]
                rec = insights['recommendations'].get(ticker, 'Hold')
                risk_level = insights['risk_levels'].get(ticker, 'Medium')
                summary += f"| {ticker} | {rec} | {risk_level} | {risk.get('sharpe', 0):.2f} |\n"
        
        summary += "\n---\n"
        summary += "*⚠️ Not financial advice. For educational purposes only.*"
        
        return summary
    
    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("History cleared")