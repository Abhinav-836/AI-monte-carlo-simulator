"""
Response parser for LLM output (Markdown-aware).
"""

import re
import textwrap
from typing import Any, Dict, List, Optional

_SECTION_KEYS = [  # longest first so "key risks" wins over "risk"
    ("executive summary", "summary"), ("individual asset", "analysis"), ("asset by asset", "analysis"),
    ("strategic insights", "insights"), ("recommendations", "recommendation"), ("recommendation", "recommendation"),
    ("key risks", "risk"), ("conclusion", "conclusion"), ("analysis", "analysis"), ("insights", "insights"),
    ("outlook", "outlook"), ("summary", "summary"), ("risks", "risk"), ("risk", "risk"),
]
_POSITIVE = ["growth", "profit", "profitable", "bullish", "strong", "upside", "outperform", "gain", "gains", "opportunity"]
_NEGATIVE = ["loss", "losses", "bearish", "decline", "downside", "underperform", "weak", "drawdown", "selloff"]
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*#\"'(])")


class ResponseParser:

    # ================= CLEAN =================
    @staticmethod
    def clean_response(response: str) -> str:
        """Normalise whitespace without breaking Markdown (indentation, code blocks, tables)."""
        if not response:
            return ""
        text = str(response).replace("\r\n", "\n").replace("\r", "\n")
        out, in_code = [], False
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                in_code = not in_code
                out.append(line.rstrip())
                continue
            if in_code:
                out.append(line.rstrip())
                continue
            m = re.match(r"^([ \t]*)(.*)$", line)
            indent, body = m.group(1).replace("\t", "    "), m.group(2)
            body = re.sub(r"[ \t]{2,}", " ", body).rstrip()
            body = re.sub(r"(?<=\w)[ \t]+([,.!?;:])(?=\s|$)", r"\1", body)
            out.append(indent + body if body else "")
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
        return text.strip()

    # ================= BULLETS =================
    @staticmethod
    def extract_bullet_points(text: str) -> List[str]:
        if not text:
            return []
        bullets = []
        for line in text.split("\n"):
            s = line.strip()
            m = (re.match(r"^[-*•●○▪]\s+(.*)$", s)
                 or re.match(r"^\d+[.)]\s+(.*)$", s)
                 or re.match(r"^[a-zA-Z][.)]\s+(.*)$", s))
            if m and m.group(1).strip():
                bullets.append(m.group(1).strip())
        return bullets

    # ================= METRICS =================
    @staticmethod
    def extract_key_metrics(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        metrics: Dict[str, Any] = {}

        pct = re.findall(r"([-+]?\d+(?:\.\d+)?)\s?%", text)
        if pct:
            metrics["percentages"] = [float(x) for x in pct]

        prices = re.findall(r"(?:HK\$|C\$|A\$|\$|₹|€|£|¥)\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", text)
        if prices:
            metrics["prices"] = [float(x.replace(",", "")) for x in prices]

        sharpe = re.findall(r"sharpe(?:\s+ratio)?\s*(?:is|of|=|:)?\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
        if sharpe:
            metrics["sharpe"] = [float(x) for x in sharpe]

        var = re.findall(
            r"\bvar\b(?:\s*\(\s*\d+\s*%\s*\))?\s*(?:is|of|=|:|at)?\s*([-+]?\d+(?:\.\d+)?)\s?%", text, re.I)
        if var:
            metrics["var"] = [float(x) for x in var]

        ci = re.findall(
            r"\[\s*[^\d\]]*?(\d[\d,]*(?:\.\d+)?)\s*(?:,|–|to)\s*[^\d\]]*?(\d[\d,]*(?:\.\d+)?)\s*\]", text)
        if ci:
            metrics["confidence_intervals"] = [
                (float(a.replace(",", "")), float(b.replace(",", ""))) for a, b in ci]
        return metrics

    # ================= SECTIONS =================
    @staticmethod
    def _header_name(line: str) -> Optional[str]:
        s = line.strip()
        if not s:
            return None
        md = re.match(r"^#{1,6}\s+(.*)$", s)
        if md:
            title, is_markdown = md.group(1), True
        elif (re.match(r"^\*\*[^*]+\*\*:?$", s) or (s.endswith(":") and len(s) <= 40)
              or (s.isupper() and len(s) <= 40)):
            title, is_markdown = s, False
        else:
            return None
        clean = " ".join(re.sub(r"[^a-z ]", " ", title.lower()).split())
        for key, name in _SECTION_KEYS:
            if key in clean:
                return name
        return (clean.replace(" ", "_") or "section") if is_markdown else None

    @staticmethod
    def structure_by_section(text: str) -> Dict[str, str]:
        if not text:
            return {}
        sections: Dict[str, str] = {}
        current, buffer = "main", []

        def flush():
            body = "\n".join(buffer).strip()
            if body:
                sections[current] = (sections[current] + "\n\n" + body) if current in sections else body

        for line in text.split("\n"):
            name = ResponseParser._header_name(line)
            if name:
                flush()
                current, buffer = name, []
            else:
                buffer.append(line)
        flush()
        return sections

    # ================= FORMAT =================
    @staticmethod
    def format_for_display(text: str, max_width: int = 80) -> str:
        if not text:
            return ""
        out, in_code = [], False
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if in_code or line.lstrip().startswith("```") or line.lstrip().startswith("|") or len(line) <= max_width:
                out.append(line)
                continue
            indent = re.match(r"^\s*", line).group(0)
            out.append(textwrap.fill(line.strip(), width=max_width, initial_indent=indent,
                                     subsequent_indent=indent, break_long_words=False, break_on_hyphens=False))
        return "\n".join(out)

    # ================= SENTIMENT =================
    @staticmethod
    def extract_sentiment(text: str) -> str:
        if not text:
            return "neutral"
        t = text.lower()
        pos = sum(len(re.findall(rf"\b{w}\b", t)) for w in _POSITIVE)
        neg = sum(len(re.findall(rf"\b{w}\b", t)) for w in _NEGATIVE)
        score = pos - neg
        if score > 2:
            return "positive"
        if score < -2:
            return "negative"
        return "neutral"

    # ================= RECOMMENDATION =================
    @staticmethod
    def get_key_recommendation(text: str) -> Optional[str]:
        if not text:
            return None
        for s in _SENTENCE_SPLIT.split(text):
            if any(k in s.lower() for k in ("recommend", "suggest", "consider", "should")):
                return s.strip()
        return None

    # ================= SUMMARY =================
    @staticmethod
    def summarize_response(text: str, max_sentences: int = 3) -> str:
        if not text:
            return ""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        return " ".join(sentences[:max_sentences])