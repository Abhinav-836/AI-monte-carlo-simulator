"""
Prompt templates for the LLM explainer (single source of truth for every prompt).
Missing / NaN values are written as "n/a" — never as a fake 0 the model would treat as data.
"""

from typing import Any, Dict, List, Optional

from src.market_utils import format_price, is_number

UNITS_NOTE = (
    "Units: prices are in each asset's local currency (LSE quotes are in pence). "
    "'Daily VaR' is the 1-day 95% loss threshold of simulated log returns. "
    "'Drift' and 'Vol' are annualised from the simulation (drift is estimated from recent history). "
    "'Sharpe' uses a 2% risk-free rate. 'Avg drawdown' is the average worst decline per simulated path."
)


class PromptTemplates:

    # ================= SAFE FORMATTERS =================
    @staticmethod
    def _num(val) -> Optional[float]:
        return float(val) if is_number(val) else None

    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        f = PromptTemplates._num(val)
        return default if f is None else f

    @staticmethod
    def _safe_pct(val) -> float:
        return PromptTemplates._safe_float(val) * 100

    @staticmethod
    def _fmt(val, spec: str = ".2f", scale: float = 1.0, suffix: str = "") -> str:
        f = PromptTemplates._num(val)
        return "n/a" if f is None else f"{f * scale:{spec}}{suffix}"

    @staticmethod
    def _pct(val, spec: str = ".1f") -> str:
        return PromptTemplates._fmt(val, spec, 100.0, "%")

    @staticmethod
    def _price(val, ticker: str) -> str:
        p = format_price(val, ticker)
        return "n/a" if p == "—" else p

    @staticmethod
    def _ci(ci, ticker: str) -> str:
        if isinstance(ci, (list, tuple)) and len(ci) >= 2:
            return f"{PromptTemplates._price(ci[0], ticker)} to {PromptTemplates._price(ci[1], ticker)}"
        return "n/a"

    @staticmethod
    def _horizon(h) -> str:
        return f"{int(float(h))} trading days" if is_number(h) and float(h) > 0 else "the simulation horizon"

    @staticmethod
    def _asset_line(t: str, expected_prices: Dict, confidence_intervals: Dict, risk_metrics: Dict) -> str:
        P = PromptTemplates
        r = (risk_metrics or {}).get(t) or {}
        return (
            f"- {t}: expected {P._price((expected_prices or {}).get(t), t)} "
            f"(95% range {P._ci((confidence_intervals or {}).get(t), t)}), "
            f"daily VaR {P._pct(r.get('var_95'), '.2f')}, drift {P._pct(r.get('expected_return'))}, "
            f"vol {P._pct(r.get('volatility'))}, Sharpe {P._fmt(r.get('sharpe'))}, "
            f"avg drawdown {P._pct(r.get('max_drawdown'))}"
        )

    @staticmethod
    def _insights_block(insights: Optional[Dict[str, Any]]) -> str:
        if not insights or not insights.get("best_performer"):
            return ""
        P = PromptTemplates
        lines = [
            f"- Overall tone: {insights.get('overall_sentiment', 'n/a')}",
            f"- Highest Sharpe: {insights['best_performer']} ({P._fmt(insights.get('best_sharpe'))}); "
            f"lowest: {insights.get('worst_performer')} ({P._fmt(insights.get('worst_sharpe'))})",
            f"- Average Sharpe {P._fmt(insights.get('average_sharpe'))}, average daily VaR "
            f"{P._pct(insights.get('average_var'), '.2f')}, average vol {P._pct(insights.get('average_volatility'))}",
        ]
        signals = insights.get("signals") or {}
        if signals:
            lines.append("- Mechanical screen: " + ", ".join(f"{t}={s}" for t, s in signals.items()))
        return "RULE-BASED SCREEN (computed mechanically from the data above):\n" + "\n".join(lines)

    # ================= SIMULATION SUMMARY =================
    @staticmethod
    def simulation_summary(
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: Optional[float] = None,   # kept for compatibility; not a model-quality metric
        horizon_days: Optional[int] = None,
    ) -> str:
        P = PromptTemplates
        lines = "\n".join(P._asset_line(t, expected_prices, confidence_intervals, risk_metrics) for t in tickers)
        return (
            f"Act as a financial analyst. Monte Carlo horizon: {P._horizon(horizon_days)}.\n\n"
            f"DATA:\n{lines}\n\n{UNITS_NOTE}\n\n"
            "Write 3 short paragraphs:\n1. Outlook\n2. Risks\n3. What the numbers suggest "
            "(educational, not advice)\n\nKeep it concise and use only the data above."
        )

    # ================= RISK ANALYSIS =================
    @staticmethod
    def risk_analysis(
        tickers: List[str],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: Optional[float] = None,
    ) -> str:
        P = PromptTemplates
        lines = []
        for t in tickers:
            r = (risk_metrics or {}).get(t) or {}
            lines.append(
                f"- {t}: daily VaR {P._pct(r.get('var_95'), '.2f')}, daily CVaR {P._pct(r.get('cvar_95'), '.2f')}, "
                f"Sharpe {P._fmt(r.get('sharpe'))}, avg drawdown {P._pct(r.get('max_drawdown'))}, "
                f"worst drawdown {P._pct(r.get('worst_drawdown'))}"
            )
        body = "\n".join(lines)
        return (
            f"Act as a risk analyst.\n\nRISK DATA:\n{body}\n\n{UNITS_NOTE}\n\n"
            "Explain:\n- Highest-risk asset\n- Best risk-adjusted return\n- Drawdown concerns\n- Risk mitigation ideas"
        )

    # ================= COMPARE =================
    @staticmethod
    def compare_assets(
        tickers: List[str],
        expected_prices: Dict[str, float],
        risk_metrics: Dict[str, Dict[str, float]],
    ) -> str:
        P = PromptTemplates
        lines = []
        for t in tickers:
            r = (risk_metrics or {}).get(t) or {}
            lines.append(
                f"- {t}: expected {P._price((expected_prices or {}).get(t), t)}, "
                f"vol {P._pct(r.get('volatility'))}, Sharpe {P._fmt(r.get('sharpe'))}"
            )
        body = "\n".join(lines)
        return (
            f"Act as a portfolio analyst.\n\nASSETS:\n{body}\n\n{UNITS_NOTE}\n\n"
            "Explain:\n- Strongest risk-adjusted profile\n- Risk vs return trade-off\n"
            "- How weights could be balanced (educational, not advice)"
        )

    # ================= QA =================
    @staticmethod
    def answer_question(
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: Optional[float],
        question: str,
        insights: Optional[Dict[str, Any]] = None,
        horizon_days: Optional[int] = None,
    ) -> str:
        P = PromptTemplates
        lines = "\n".join(P._asset_line(t, expected_prices, confidence_intervals, risk_metrics) for t in tickers)
        screen = P._insights_block(insights)
        return (
            f"Use ONLY this Monte Carlo data (horizon: {P._horizon(horizon_days)}):\n\n{lines}\n\n"
            f"{UNITS_NOTE}\n\n{screen}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer clearly. If the data cannot answer it, say \"Insufficient data\" and explain what is missing."
        )

    # ================= SCENARIO =================
    @staticmethod
    def scenario_analysis(
        tickers: List[str],
        base_results: Dict[str, Any],
        scenario_results: Dict[str, Any],
        scenario_description: str,
    ) -> str:
        P = PromptTemplates
        base_exp = (base_results or {}).get("expected_prices", {})
        scen_exp = (scenario_results or {}).get("expected_prices", {})
        lines = []
        for t in tickers:
            b, s = P._num(base_exp.get(t)), P._num(scen_exp.get(t))
            change = f"{(s / b - 1) * 100:+.1f}%" if b and s is not None else "n/a"
            lines.append(f"- {t}: {P._price(b, t)} -> {P._price(s, t)} ({change})")
        body = "\n".join(lines)
        return (
            f"Scenario: {scenario_description}\n\n{body}\n\n"
            "Explain:\n- Impact on expected prices\n- Risk change\n- Key insight"
        )

    # ================= EXECUTIVE =================
    @staticmethod
    def executive_summary(
        tickers: List[str],
        expected_prices: Dict[str, float],
        risk_metrics: Dict[str, Dict[str, float]],
        variance_reduction: Optional[float] = None,
    ) -> str:
        P = PromptTemplates
        lines = []
        for t in tickers:
            r = (risk_metrics or {}).get(t) or {}
            lines.append(f"- {t}: {P._price((expected_prices or {}).get(t), t)}, "
                         f"Sharpe {P._fmt(r.get('sharpe'))}, daily VaR {P._pct(r.get('var_95'), '.2f')}")
        body = "\n".join(lines)
        return (
            f"Executive summary data:\n\n{body}\n\n{UNITS_NOTE}\n\n"
            "Give:\n- Key takeaway\n- Biggest risk\n- One thing to watch"
        )

    # ================= COMPREHENSIVE (used by the explainer) =================
    @staticmethod
    def comprehensive_analysis(
        tickers: List[str],
        expected_prices: Dict[str, float],
        confidence_intervals: Dict[str, List[float]],
        risk_metrics: Dict[str, Dict[str, float]],
        insights: Optional[Dict[str, Any]] = None,
        horizon_days: Optional[int] = None,
    ) -> str:
        P = PromptTemplates
        lines = "\n".join(P._asset_line(t, expected_prices, confidence_intervals, risk_metrics) for t in tickers)
        screen = P._insights_block(insights)
        return (
            f"Analyse this Monte Carlo simulation (horizon: {P._horizon(horizon_days)}).\n\n"
            f"ASSET DATA:\n{lines}\n\n{UNITS_NOTE}\n\n{screen}\n\n"
            "Respond in Markdown with these sections:\n\n"
            "## 📊 Executive summary\n(2-3 sentences)\n\n"
            "## 📈 Asset by asset\n(2-3 lines per asset: expected move, range, risk)\n\n"
            "## ⚠️ Key risks\n(3-4 bullets, including model limitations)\n\n"
            "## 💡 Insights\n(2-3 bullets)\n\n"
            "Be concise and quantitative. Do not invent numbers that are not in the data."
        )