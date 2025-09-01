"""
Setup script for Multi-Agent RAG System

Run this to set up the system for first use.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/pdfs",
        "data/vector_store",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_config():
    """Set up configuration files"""
    config_dir = Path("config")
    env_template = Path("env.template")
    env_file = Path(".env")
    config_file = config_dir / "config.py"
    
    # Create .env file if it doesn't exist
    if not env_file.exists():
        if env_template.exists():
            # Copy template to .env
            with open(env_template, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env and add your API keys!")
        else:
            print("❌ env.template not found")
            return False
    else:
        print("⚠️  .env file already exists")
    
    # Create config.py if it doesn't exist (optional)
    if not config_file.exists():
        example_file = config_dir / "config_example.py"
        if example_file.exists():
            # Copy example to config.py
            with open(example_file, 'r') as f:
                content = f.read()
            
            with open(config_file, 'w') as f:
                f.write(content)
            
            print("✅ Created config.py from example")
        else:
            print("⚠️  config_example.py not found, using default config")
    
    return True


def check_api_keys():
    """Check if API keys are configured"""
    try:
        sys.path.insert(0, str(Path.cwd()))
        from config import CONFIG
        
        llm = CONFIG["llm"]
        
        providers = {
            "OpenAI": llm.openai_api_key and llm.openai_api_key.startswith("sk-") and len(llm.openai_api_key) > 20,
            "Anthropic": llm.anthropic_api_key and len(llm.anthropic_api_key or "") > 20,
            "Google Gemini": llm.google_api_key and len(llm.google_api_key or "") > 20
        }
        
        configured_providers = [name for name, configured in providers.items() if configured]
        
        if configured_providers:
            print(f"✅ API keys configured for: {', '.join(configured_providers)}")
            return True
        else:
            print("⚠️  No API keys detected")
            print("Please edit .env file and add at least one API key")
            
            # Check if .env file exists
            env_file = Path(".env")
            if env_file.exists():
                print(f"📄 .env file found at: {env_file}")
                print("   Edit this file to add your API keys")
            else:
                print("📄 .env file not found")
                print("   Run setup again or manually copy: cp env.template .env")
            
            return False
    
    except ImportError as e:
        print(f"⚠️  Could not check API keys: {e}")
        print("Make sure you have run: pip install -r requirements.txt")
        return False


def run_quick_test():
    """Run a quick test of the system"""
    print("\n🧪 Running quick system test...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from main import MultiAgentRAGSystem
        
        # Test initialization
        system = MultiAgentRAGSystem()
        system.initialize()
        
        # Get status
        status = system.get_system_status()
        print(f"✅ System initialized with {status['manager']['total_agents']} agents")
        
        # Quick test query
        response = system.query("Hello, can you tell me what you can do?")
        print("✅ System responded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Multi-Agent RAG System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return False
    
    # Setup configuration
    if not setup_config():
        print("❌ Setup failed during configuration")
        return False
    
    # Check API keys
    api_keys_configured = check_api_keys()
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    print("✅ Directories created")
    print("✅ Dependencies installed")
    print("✅ Configuration file created")
    
    if api_keys_configured:
        print("✅ API keys configured")
        
        # Run test if API keys are configured
        test_passed = run_quick_test()
        if test_passed:
            print("✅ System test passed")
        else:
            print("⚠️  System test failed")
    else:
        print("⚠️  API keys not configured")
    
    print("\n" + "=" * 40)
    print("🎉 Setup Complete!")
    
    if not api_keys_configured:
        print("\n📝 Next Steps:")
        print("1. Edit .env file and add your API keys:")
        print("   nano .env  # or use your preferred editor")
        print("2. Add at least one API key (OpenAI, Anthropic, or Google Gemini)")
        print("3. Run: python main.py")
        print("\n📝 API Key Sources:")
        print("- OpenAI: https://platform.openai.com/api-keys")
        print("- Anthropic: https://console.anthropic.com/")
        print("- Google Gemini: https://makersuite.google.com/app/apikey")
    else:
        print("\n🚀 Ready to use! Run: python main.py")
    
    print("\n📚 Add PDF files to data/pdfs/ to use the document retrieval feature")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
