"""
Example configuration for the Multi-Agent RAG System using .env files

This shows how to use environment variables for secure configuration.
The system automatically loads from .env files using python-dotenv.
"""

import os
from pathlib import Path
from config_template import DEFAULT_CONFIG, LLMConfig, RAGConfig, AgentConfig

# =============================================================================
# RECOMMENDED APPROACH: Use .env file
# =============================================================================

# 1. Copy env.template to .env:
#    cp env.template .env
#
# 2. Edit .env with your actual API keys:
#    OPENAI_API_KEY=sk-your-actual-key-here
#    ANTHROPIC_API_KEY=your-actual-key-here
#    GOOGLE_API_KEY=your-actual-key-here
#
# 3. The system will automatically load these values!

# The configuration will be loaded automatically from .env
# You don't need to modify this file if using .env approach
CONFIG = DEFAULT_CONFIG

# =============================================================================
# ALTERNATIVE: Manual override (not recommended for production)
# =============================================================================

# If you want to override specific values programmatically:
# CONFIG = {
#     "llm": LLMConfig(
#         openai_api_key="sk-your-key-here",  # Override from .env
#         default_provider="openai"
#     ),
#     "rag": RAGConfig(
#         chunk_size=1024,  # Override default
#         max_retrieval_docs=10
#     ),
#     "agent": AgentConfig(
#         verbose=False,  # Override default
#         enable_text_generation=False
#     )
# }

# =============================================================================
# Validation and debugging functions
# =============================================================================

def validate_config():
    """Validate that at least one LLM provider is configured"""
    llm = CONFIG["llm"]
    
    providers_available = {
        "openai": bool(llm.openai_api_key and llm.openai_api_key.startswith("sk-")),
        "anthropic": bool(llm.anthropic_api_key and len(llm.anthropic_api_key or "") > 20),
        "gemini": bool(llm.google_api_key and len(llm.google_api_key or "") > 20)
    }
    
    if not any(providers_available.values()):
        print("⚠️  Warning: No LLM providers configured with valid API keys!")
        print("Please create .env file with your API keys (copy from env.template)")
        print(f"Provider status: {providers_available}")
        return False
    
    configured_providers = [k for k, v in providers_available.items() if v]
    print(f"✅ Configuration validated. Available providers: {configured_providers}")
    return True


def show_config_status():
    """Show current configuration status"""
    print("📋 Configuration Status:")
    print(f"  Default LLM Provider: {CONFIG['llm'].default_provider}")
    print(f"  PDF Directory: {CONFIG['rag'].pdf_directory}")
    print(f"  Chunk Size: {CONFIG['rag'].chunk_size}")
    print(f"  Max Retrieval Docs: {CONFIG['rag'].max_retrieval_docs}")
    print(f"  Web Search Enabled: {CONFIG['agent'].enable_web_search}")
    print(f"  Text Generation Enabled: {CONFIG['agent'].enable_text_generation}")
    print(f"  Verbose Mode: {CONFIG['agent'].verbose}")
    
    # Check .env file
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        print(f"  .env file: Found at {env_file}")
    else:
        print(f"  .env file: Not found (expected at {env_file})")
        print("  To create: cp env.template .env")


def get_setup_instructions():
    """Get setup instructions for configuration"""
    instructions = """
🔧 Configuration Setup Instructions:

1. Create .env file:
   cp env.template .env

2. Edit .env file with your API keys:
   nano .env  # or use your preferred editor

3. Add at least one API key:
   OPENAI_API_KEY=sk-your-actual-openai-key
   # OR
   ANTHROPIC_API_KEY=your-actual-anthropic-key
   # OR  
   GOOGLE_API_KEY=your-actual-gemini-key

4. Optional: Customize other settings in .env

5. Run the system:
   python main.py

📝 Getting API Keys:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/
- Google Gemini: https://makersuite.google.com/app/apikey

🔒 Security Note:
- .env files are automatically ignored by git
- Never commit API keys to version control
- Keep your .env file secure and local only
"""
    return instructions


# Uncomment to validate configuration on import
# validate_config()
