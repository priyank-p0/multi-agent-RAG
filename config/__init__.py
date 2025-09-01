"""Configuration module for Multi-Agent RAG System"""

from .config_template import DEFAULT_CONFIG, LLMConfig, RAGConfig, AgentConfig

try:
    from .config import CONFIG
except ImportError:
    # Fallback to default config if user hasn't created config.py
    CONFIG = DEFAULT_CONFIG
    print("Using default configuration. Copy config_template.py to config.py and update with your API keys.")

__all__ = ["CONFIG", "LLMConfig", "RAGConfig", "AgentConfig"]
