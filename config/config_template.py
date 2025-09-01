"""
Configuration template for the Multi-Agent RAG System

This template loads configuration from environment variables (.env file).
Copy this file to config.py if you want to customize it.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not installed. Using system environment variables only.")
    print("   Install with: pip install python-dotenv")

def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_int_env(key: str, default: int) -> int:
    """Get integer value from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class LLMConfig:
    """Configuration for different LLM providers"""
    # API Keys - Loaded from .env file or environment variables
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Model configurations
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    
    # Default provider (openai, anthropic, or gemini)
    default_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Filter out placeholder values
        if self.openai_api_key == "your_openai_api_key_here":
            self.openai_api_key = None
        if self.anthropic_api_key == "your_anthropic_api_key_here":
            self.anthropic_api_key = None
        if self.google_api_key == "your_google_gemini_api_key_here":
            self.google_api_key = None
        if self.huggingface_token == "your_huggingface_token_here":
            self.huggingface_token = None

@dataclass
class RAGConfig:
    """Configuration for RAG components"""
    # PDF processing
    pdf_directory: str = os.getenv("PDF_DIRECTORY", "data/pdfs")
    
    # Embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Chunking parameters
    chunk_size: int = get_int_env("CHUNK_SIZE", 512)
    chunk_overlap: int = get_int_env("CHUNK_OVERLAP", 50)
    
    # Retrieval parameters
    max_retrieval_docs: int = get_int_env("MAX_RETRIEVAL_DOCS", 7)
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Vector store settings
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")

@dataclass
class AgentConfig:
    """Configuration for agent behavior"""
    max_iterations: int = get_int_env("MAX_ITERATIONS", 4)
    verbose: bool = get_bool_env("VERBOSE", True)
    enable_web_search: bool = get_bool_env("ENABLE_WEB_SEARCH", True)
    enable_text_generation: bool = get_bool_env("ENABLE_TEXT_GENERATION", True)

# Default configuration instance
DEFAULT_CONFIG = {
    "llm": LLMConfig(),
    "rag": RAGConfig(),
    "agent": AgentConfig()
}
