#!/usr/bin/env python3
"""
Demo script showing how to use .env configuration

This script demonstrates the new environment variable configuration approach.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_demo_env():
    """Create a demo .env file for testing"""
    demo_env_content = """# Demo .env file for Multi-Agent RAG System
# This is for demonstration only - replace with your actual API keys

# LLM API Keys (replace with real keys)
OPENAI_API_KEY=sk-demo-key-replace-with-real-key
ANTHROPIC_API_KEY=demo-anthropic-key-replace-with-real-key
GOOGLE_API_KEY=demo-google-key-replace-with-real-key

# Configuration
DEFAULT_LLM_PROVIDER=openai
CHUNK_SIZE=256
MAX_RETRIEVAL_DOCS=5
ENABLE_WEB_SEARCH=true
ENABLE_IMAGE_GENERATION=false
VERBOSE=true
"""
    
    demo_path = Path(".env.demo")
    with open(demo_path, 'w') as f:
        f.write(demo_env_content)
    
    print(f"✅ Created demo .env file: {demo_path}")
    return demo_path

def test_env_loading():
    """Test loading configuration from environment variables"""
    print("🧪 Testing environment variable loading...")
    
    # Load the configuration
    try:
        from config import CONFIG
        
        print("\n📋 Current Configuration:")
        print(f"  LLM Provider: {CONFIG['llm'].default_provider}")
        print(f"  OpenAI Key: {'✅ Set' if CONFIG['llm'].openai_api_key else '❌ Not set'}")
        print(f"  Anthropic Key: {'✅ Set' if CONFIG['llm'].anthropic_api_key else '❌ Not set'}")
        print(f"  Google Key: {'✅ Set' if CONFIG['llm'].google_api_key else '❌ Not set'}")
        print(f"  PDF Directory: {CONFIG['rag'].pdf_directory}")
        print(f"  Chunk Size: {CONFIG['rag'].chunk_size}")
        print(f"  Max Retrieval Docs: {CONFIG['rag'].max_retrieval_docs}")
        print(f"  Web Search: {CONFIG['agent'].enable_web_search}")
        print(f"  Image Generation: {CONFIG['agent'].enable_image_generation}")
        print(f"  Verbose: {CONFIG['agent'].verbose}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def show_setup_instructions():
    """Show setup instructions for .env configuration"""
    instructions = """
🔧 .env Configuration Setup:

1. Create .env file:
   cp env.template .env

2. Edit .env with your API keys:
   nano .env

3. Required environment variables:
   OPENAI_API_KEY=sk-your-actual-key
   # OR
   ANTHROPIC_API_KEY=your-actual-key
   # OR
   GOOGLE_API_KEY=your-actual-key

4. Optional customizations:
   DEFAULT_LLM_PROVIDER=openai
   CHUNK_SIZE=512
   MAX_RETRIEVAL_DOCS=7
   ENABLE_WEB_SEARCH=true
   ENABLE_IMAGE_GENERATION=true
   VERBOSE=true

5. Run the system:
   python main.py

🔒 Security Benefits:
- .env files are ignored by git (never committed)
- API keys stay local and secure
- Easy to manage different environments
- No hardcoded secrets in code

📝 Getting API Keys:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/
- Google Gemini: https://makersuite.google.com/app/apikey
"""
    return instructions

def main():
    """Main demo function"""
    print("🌟 Multi-Agent RAG System - .env Configuration Demo")
    print("=" * 60)
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found: {env_file}")
    else:
        print(f"⚠️  .env file not found: {env_file}")
        if Path("env.template").exists():
            print("💡 You can create one by running: cp env.template .env")
        else:
            print("❌ env.template also not found")
    
    # Test configuration loading
    print("\n" + "=" * 60)
    success = test_env_loading()
    
    if not success:
        print("\n" + "=" * 60)
        print("❌ Configuration loading failed")
        print("This is likely because:")
        print("1. .env file doesn't exist")
        print("2. python-dotenv not installed")
        print("3. Missing dependencies")
        
        print("\n🔧 Quick fix:")
        print("pip install python-dotenv")
        print("cp env.template .env")
        print("# Edit .env with your API keys")
    
    # Show setup instructions
    print("\n" + "=" * 60)
    print(show_setup_instructions())
    
    # Create demo file
    print("=" * 60)
    print("Creating demo .env file for reference...")
    demo_path = create_demo_env()
    print(f"📄 Check {demo_path} for an example configuration")
    print("⚠️  Remember to replace demo keys with real API keys!")

if __name__ == "__main__":
    main()
