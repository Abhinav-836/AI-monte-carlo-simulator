"""
Prompt templates (FIXED + OPTIMIZED for small LLMs like LLaMA 3.1 / gpt-oss)
"""

from typing import Dict, Any, List


class PromptTemplates:
    """Optimized prompt templates for small LLM performance"""

    # ================= SAFE FORMATTERS =================
    @staticmethod
    def _safe_float(val, default=0.0):
        try:
            return float(val)
        except:
            return default

    @staticmethod
    def _safe_pct(val):
        return PromptTemplates._safe_float(val) * 100

    # ================= SIMULATION SUMMARY =================
    @staticmethod
    def simulation_summary(
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float
    ) -> str:

        ticker_section = ""

        for ticker in tickers:
            price = PromptTemplates._safe_float(expected_prices.get(ticker))
            ci = confidence_intervals.get(ticker, [0, 0])

            ci_low = PromptTemplates._safe_float(ci[0] if len(ci) > 0 else 0)
            ci_high = PromptTemplates._safe_float(ci[1] if len(ci) > 1 else 0)

            risk = risk_metrics.get(ticker, {})

            ticker_section += (
                f"{ticker}: "
                f"Price=${price:.2f}, "
                f"CI=[{ci_low:.2f},{ci_high:.2f}], "
                f"VaR={PromptTemplates._safe_pct(risk.get('var_95')):.1f}%, "
                f"Return={PromptTemplates._safe_pct(risk.get('expected_return')):.1f}%, "
                f"Sharpe={PromptTemplates._safe_float(risk.get('sharpe')):.2f}\n"
            )

        return f"""Act as a financial analyst.

DATA:
{ticker_section}

Model variance reduction: {variance_reduction*100:.1f}%

Write 3 short paragraphs:
1. Outlook
2. Risks
3. Recommendation

Keep it concise and professional.
"""

    # ================= RISK ANALYSIS =================
    @staticmethod
    def risk_analysis(
        tickers: List[str],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float
    ) -> str:

        lines = []

        for ticker in tickers:
            r = risk_metrics.get(ticker, {})
            lines.append(
                f"{ticker}: "
                f"VaR={PromptTemplates._safe_pct(r.get('var_95')):.1f}%, "
                f"CVaR={PromptTemplates._safe_pct(r.get('cvar_95')):.1f}%, "
                f"Sharpe={PromptTemplates._safe_float(r.get('sharpe')):.2f}, "
                f"DD={PromptTemplates._safe_pct(r.get('max_drawdown')):.1f}%"
            )

        return f"""Act as a risk analyst.

RISK DATA:
{chr(10).join(lines)}

Variance reduction: {variance_reduction*100:.1f}%

Explain:
- Highest risk asset
- Best risk-adjusted return
- Drawdown concerns
- Risk mitigation
"""

    # ================= COMPARE =================
    @staticmethod
    def compare_assets(
        tickers: List[str],
        expected_prices: Dict[str, float],
        risk_metrics: Dict[str, Dict[str, float]]
    ) -> str:

        lines = []

        for ticker in tickers:
            exp = PromptTemplates._safe_float(expected_prices.get(ticker))
            r = risk_metrics.get(ticker, {})

            lines.append(
                f"{ticker}: Price={exp:.2f}, "
                f"Vol={PromptTemplates._safe_pct(r.get('volatility')):.1f}%, "
                f"Sharpe={PromptTemplates._safe_float(r.get('sharpe')):.2f}"
            )

        return f"""Act as a portfolio manager.

ASSETS:
{chr(10).join(lines)}

Explain:
- Best asset
- Risk vs return tradeoff
- Allocation suggestion
"""

    # ================= QA =================
    @staticmethod
    def answer_question(
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float,
        question: str
    ) -> str:

        summary = []

        for t in tickers:
            price = PromptTemplates._safe_float(expected_prices.get(t))
            ci = confidence_intervals.get(t, [0, 0])

            summary.append(
                f"{t}: Price={price:.2f}, "
                f"CI=[{ci[0]:.2f},{ci[1]:.2f}], "
                f"VaR={PromptTemplates._safe_pct(risk_metrics.get(t, {}).get('var_95')):.1f}%"
            )

        return f"""Use ONLY this data:

{chr(10).join(summary)}

Question: {question}

Answer clearly. If unknown, say "Insufficient data".
"""

    # ================= SCENARIO =================
    @staticmethod
    def scenario_analysis(
        tickers: List[str],
        base_results: Dict[str, Any],
        scenario_results: Dict[str, Any],
        scenario_description: str
    ) -> str:

        base_exp = base_results.get('expected_prices', {})
        scen_exp = scenario_results.get('expected_prices', {})

        comparison = []

        for ticker in tickers:
            b = PromptTemplates._safe_float(base_exp.get(ticker))
            s = PromptTemplates._safe_float(scen_exp.get(ticker))

            change_pct = ((s - b) / b * 100) if b != 0 else 0

            comparison.append(
                f"{ticker}: {b:.2f} → {s:.2f} ({change_pct:+.1f}%)"
            )

        return f"""Scenario: {scenario_description}

{chr(10).join(comparison)}

Explain:
- Impact on returns
- Risk change
- Key insight
"""

    # ================= EXECUTIVE =================
    @staticmethod
    def executive_summary(
        tickers: List[str],
        expected_prices: Dict[str, float],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: float
    ) -> str:

        bullets = []

        for ticker in tickers:
            exp = PromptTemplates._safe_float(expected_prices.get(ticker))
            sharpe = PromptTemplates._safe_float(risk_metrics.get(ticker, {}).get('sharpe'))
            var = PromptTemplates._safe_pct(risk_metrics.get(ticker, {}).get('var_95'))

            bullets.append(
                f"{ticker}: ${exp:.2f}, Sharpe={sharpe:.2f}, VaR={var:.1f}%"
            )

        return f"""Executive summary:

{chr(10).join(bullets)}

Variance reduction: {variance_reduction*100:.1f}%

Give:
- Key takeaway
- Biggest risk
- One action
"""
