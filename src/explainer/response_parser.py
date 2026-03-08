"""
Response Parser (FIXED + PRODUCTION READY)
Optimized for LLaMA / Ollama outputs
"""

import re
from typing import Dict, Any, List, Optional


class ResponseParser:
    """Robust parser for LLM outputs"""

    # ================= CLEAN =================
    @staticmethod
    def clean_response(response: str) -> str:
        if not response:
            return ""

        response = str(response)

        # Normalize newlines
        response = re.sub(r'\n{3,}', '\n\n', response)

        # Remove extra spaces
        response = re.sub(r'[ \t]+', ' ', response)

        # Fix punctuation spacing
        response = re.sub(r'\s+([.,!?;:])', r'\1', response)

        # Trim lines
        response = '\n'.join(line.rstrip() for line in response.split('\n'))

        response = response.strip()

        # Ensure proper ending
        if response and response[-1] not in '.!?':
            response += '.'

        return response

    # ================= BULLETS =================
    @staticmethod
    def extract_bullet_points(text: str) -> List[str]:
        if not text:
            return []

        bullets = []

        lines = text.split('\n')
        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Bullet symbols
            if re.match(r'^[•\-\*\u25CF\u25CB]\s+', line):
                bullets.append(line[1:].strip())

            # Numbered lists
            elif re.match(r'^\d+\.\s+', line):
                bullets.append(re.sub(r'^\d+\.\s+', '', line))

            # Letter lists
            elif re.match(r'^[a-zA-Z]\.\s+', line):
                bullets.append(re.sub(r'^[a-zA-Z]\.\s+', '', line))

        return bullets

    # ================= METRICS =================
    @staticmethod
    def extract_key_metrics(text: str) -> Dict[str, Any]:
        if not text:
            return {}

        metrics = {}

        # Percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)%', text)
        if percentages:
            metrics['percentages'] = list(map(float, percentages))

        # Dollar values
        dollars = re.findall(r'\$(\d+(?:\.\d+)?)', text)
        if dollars:
            metrics['dollars'] = list(map(float, dollars))

        # Sharpe
        sharpe = re.findall(r'sharpe[:\s]*([-+]?\d+(?:\.\d+)?)', text, re.I)
        if sharpe:
            metrics['sharpe'] = list(map(float, sharpe))

        # VaR
        var = re.findall(r'var[:\s]*([-+]?\d+(?:\.\d+)?)%', text, re.I)
        if var:
            metrics['var'] = list(map(float, var))

        # Confidence intervals
        ci = re.findall(r'\[\$?(\d+(?:\.\d+)?),\s*\$?(\d+(?:\.\d+)?)\]', text)
        if ci:
            metrics['confidence_intervals'] = [(float(a), float(b)) for a, b in ci]

        return metrics

    # ================= SECTIONS =================
    @staticmethod
    def structure_by_section(text: str) -> Dict[str, str]:
        if not text:
            return {}

        sections = {}
        current_section = "main"
        buffer = []

        headers = {
            "SUMMARY": "summary",
            "ANALYSIS": "analysis",
            "RISK": "risk",
            "RECOMMENDATION": "recommendation",
            "CONCLUSION": "conclusion",
            "INSIGHTS": "insights",
            "OUTLOOK": "outlook"
        }

        lines = text.split("\n")

        for line in lines:
            stripped = line.strip().upper()

            found = False
            for key, name in headers.items():
                if stripped.startswith(key):
                    if buffer:
                        sections[current_section] = "\n".join(buffer).strip()
                    current_section = name
                    buffer = []
                    found = True
                    break

            if not found:
                buffer.append(line)

        if buffer:
            sections[current_section] = "\n".join(buffer).strip()

        return sections

    # ================= FORMAT =================
    @staticmethod
    def format_for_display(text: str, max_width: int = 80) -> str:
        if not text:
            return ""

        lines_out = []

        for line in text.split("\n"):
            if len(line) <= max_width:
                lines_out.append(line)
                continue

            words = line.split()
            current = ""

            for word in words:
                if len(current) + len(word) + 1 <= max_width:
                    current += " " + word if current else word
                else:
                    lines_out.append(current)
                    current = word

            if current:
                lines_out.append(current)

        return "\n".join(lines_out)

    # ================= SENTIMENT =================
    @staticmethod
    def extract_sentiment(text: str) -> str:
        if not text:
            return "neutral"

        text = text.lower()

        positive = sum(text.count(w) for w in [
            "growth", "profit", "bullish", "strong", "buy", "opportunity"
        ])

        negative = sum(text.count(w) for w in [
            "risk", "loss", "bearish", "decline", "sell", "weak"
        ])

        score = positive - negative

        if score > 2:
            return "positive"
        elif score < -2:
            return "negative"
        return "neutral"

    # ================= RECOMMENDATION =================
    @staticmethod
    def get_key_recommendation(text: str) -> Optional[str]:
        if not text:
            return None

        sentences = re.split(r'(?<!\d)[.!?]\s+', text)

        for s in sentences:
            if any(k in s.lower() for k in [
                "recommend", "suggest", "should", "buy", "sell", "hold"
            ]):
                return s.strip()

        return None

    # ================= SUMMARY =================
    @staticmethod
    def summarize_response(text: str, max_sentences: int = 3) -> str:
        if not text:
            return ""

        sentences = re.split(r'(?<!\d)[.!?]\s+', text)

        sentences = [s.strip() for s in sentences if s.strip()]

        return " ".join(sentences[:max_sentences])
