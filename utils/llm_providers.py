"""
LLM Provider utilities for different API providers
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from config import CONFIG


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def get_model(self) -> Any:
        """Return the model instance for smolagents"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if not openai:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.api_key = api_key or CONFIG["llm"].openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.model_name = model_name or CONFIG["llm"].openai_model
        self.client = OpenAI(api_key=self.api_key)
    
    def get_model(self):
        """Get OpenAI model for smolagents"""
        try:
            from smolagents import OpenAIServerModel
            return OpenAIServerModel(
                model_name=self.model_name,
                api_key=self.api_key
            )
        except ImportError:
            # Fallback if smolagents doesn't have OpenAIServerModel
            from smolagents import LiteLLMModel
            return LiteLLMModel(
                model_name=f"openai/{self.model_name}",
                api_key=self.api_key
            )


class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if not anthropic:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        self.api_key = api_key or CONFIG["llm"].anthropic_api_key
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.model_name = model_name or CONFIG["llm"].anthropic_model
        self.client = Anthropic(api_key=self.api_key)
    
    def get_model(self):
        """Get Anthropic model for smolagents"""
        try:
            from smolagents import LiteLLMModel
            return LiteLLMModel(
                model_name=f"anthropic/{self.model_name}",
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError("LiteLLM integration not available in smolagents")


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if not genai:
            raise ImportError("Google GenerativeAI package not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or CONFIG["llm"].google_api_key
        if not self.api_key:
            raise ValueError("Google API key not provided")
        
        self.model_name = model_name or CONFIG["llm"].gemini_model
        genai.configure(api_key=self.api_key)
    
    def get_model(self):
        """Get Gemini model for smolagents"""
        try:
            from smolagents import LiteLLMModel
            return LiteLLMModel(
                model_name=f"gemini/{self.model_name}",
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError("LiteLLM integration not available in smolagents")


def get_llm_model(provider: str = None, **kwargs) -> Any:
    """
    Get LLM model based on provider
    
    Args:
        provider: Provider name ("openai", "anthropic", "gemini")
        **kwargs: Additional arguments for the provider
    
    Returns:
        Model instance for smolagents
    """
    provider = provider or CONFIG["llm"].default_provider
    
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    provider_class = providers[provider]
    provider_instance = provider_class(**kwargs)
    return provider_instance.get_model()


def get_available_providers() -> Dict[str, bool]:
    """Check which providers are available based on installed packages and API keys"""
    available = {}
    
    # Check OpenAI
    try:
        if openai and CONFIG["llm"].openai_api_key:
            available["openai"] = True
        else:
            available["openai"] = False
    except:
        available["openai"] = False
    
    # Check Anthropic
    try:
        if anthropic and CONFIG["llm"].anthropic_api_key:
            available["anthropic"] = True
        else:
            available["anthropic"] = False
    except:
        available["anthropic"] = False
    
    # Check Gemini
    try:
        if genai and CONFIG["llm"].google_api_key:
            available["gemini"] = True
        else:
            available["gemini"] = False
    except:
        available["gemini"] = False
    
    return available
