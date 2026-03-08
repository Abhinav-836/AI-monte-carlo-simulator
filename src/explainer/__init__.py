"""
LLM Explainer module - LLaMA 3.1 1B integration
"""

from .ollama_wrapper import LlamaExplainer
from .prompt_templates import PromptTemplates
from .response_parser import ResponseParser

__all__ = [
    'LlamaExplainer',
    'PromptTemplates',
    'ResponseParser'
]