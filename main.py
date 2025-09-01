"""
Multi-Agent RAG System - Main Interface

This is the main entry point for the multi-agent RAG system.
Run this file to start the interactive interface.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.manager_agent import create_manager_agent
from utils.llm_providers import get_available_providers
from utils.pdf_processor import PDFProcessor
from config import CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_agent_rag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class MultiAgentRAGSystem:
    """Main interface for the Multi-Agent RAG System"""
    
    def __init__(self, 
                 llm_provider: str = None,
                 enable_web_search: bool = True,
                 enable_text_generation: bool = True,
                 pdf_directory: str = None):
        """
        Initialize the Multi-Agent RAG System
        
        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", "gemini")
            enable_web_search: Enable web search capabilities
            enable_text_generation: Enable comprehensive text generation capabilities
            pdf_directory: Directory containing PDF files
        """
        self.llm_provider = llm_provider
        self.enable_web_search = enable_web_search
        self.enable_text_generation = enable_text_generation
        self.pdf_directory = pdf_directory or CONFIG["rag"].pdf_directory
        
        # Check available providers
        self.available_providers = get_available_providers()
        logger.info(f"Available LLM providers: {self.available_providers}")
        
        # Select LLM provider
        if not self.llm_provider:
            self.llm_provider = self._select_provider()
        
        self.manager_agent = None
        self.pdf_processor = None
        
    def _select_provider(self) -> str:
        """Select the best available LLM provider"""
        # Priority order
        provider_priority = [CONFIG["llm"].default_provider, "openai", "anthropic", "gemini"]
        
        for provider in provider_priority:
            if self.available_providers.get(provider, False):
                logger.info(f"Selected LLM provider: {provider}")
                return provider
        
        # If no provider available, raise error
        raise RuntimeError(
            "No LLM providers available. Please configure API keys in config/config.py\n"
            f"Available providers: {list(self.available_providers.keys())}\n"
            f"Provider status: {self.available_providers}"
        )
    
    def initialize(self):
        """Initialize the system components"""
        logger.info("Initializing Multi-Agent RAG System...")
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(pdf_directory=self.pdf_directory)
        
        # Initialize manager agent
        self.manager_agent = create_manager_agent(
            llm_provider=self.llm_provider,
            enable_web_search=self.enable_web_search,
            enable_text_generation=self.enable_text_generation,
            pdf_directory=self.pdf_directory
        )
        
        logger.info("System initialized successfully")
    
    def query(self, question: str) -> str:
        """
        Process a user query
        
        Args:
            question: User question or request
            
        Returns:
            Response from the system
        """
        if not self.manager_agent:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        return self.manager_agent.run(question)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        if not self.manager_agent:
            return {"status": "not_initialized"}
        
        status = self.manager_agent.get_agent_status()
        
        # Add PDF information
        pdf_count = self.pdf_processor.get_pdf_count()
        pdf_files = self.pdf_processor.list_pdf_files()
        
        status["pdf_knowledge_base"] = {
            "directory": str(self.pdf_directory),
            "pdf_count": pdf_count,
            "pdf_files": pdf_files[:5],  # Show first 5 files
            "total_files": len(pdf_files)
        }
        
        # Add provider information
        status["llm_providers"] = {
            "selected": self.llm_provider,
            "available": self.available_providers
        }
        
        return status
    
    def add_pdfs_info(self) -> str:
        """Get information about adding PDFs to the system"""
        pdf_dir = Path(self.pdf_directory)
        
        info = f"""
        📁 PDF Knowledge Base Directory: {pdf_dir}
        
        To add PDF documents to your knowledge base:
        1. Copy your PDF files to: {pdf_dir}
        2. The system will automatically process them on the next query
        3. Supported formats: PDF files (.pdf extension)
        
        Current status:
        - PDF files found: {self.pdf_processor.get_pdf_count()}
        - Directory exists: {pdf_dir.exists()}
        
        Example usage:
        cp your_document.pdf {pdf_dir}/
        """
        
        return info.strip()
    
    def test_system(self) -> Dict[str, str]:
        """Test all system components"""
        if not self.manager_agent:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        return self.manager_agent.test_agents()


def interactive_mode():
    """Run the system in interactive mode"""
    print("🤖 Multi-Agent RAG System")
    print("=" * 40)
    
    try:
        # Initialize system
        system = MultiAgentRAGSystem()
        system.initialize()
        
        # Show system status
        status = system.get_system_status()
        print(f"\n✅ System initialized successfully!")
        print(f"📊 LLM Provider: {status['llm_providers']['selected']}")
        print(f"🤖 Active Agents: {status['manager']['total_agents']}")
        print(f"📄 PDF Files: {status['pdf_knowledge_base']['pdf_count']}")
        
        # Show PDF information
        if status['pdf_knowledge_base']['pdf_count'] == 0:
            print("\n⚠️  No PDF files found in knowledge base.")
            print(system.add_pdfs_info())
        
        print("\n" + "=" * 40)
        print("You can now ask questions! Type 'quit' to exit.")
        print("Examples:")
        print("- 'What's the latest news about AI?'")
        print("- 'Search my documents for machine learning'")
        print("- 'Give me a comprehensive analysis of quantum computing'")
        print("=" * 40)
        
        # Interactive loop
        while True:
            try:
                user_input = input("\n🧠 Ask me anything: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() in ['status', 'info']:
                    status = system.get_system_status()
                    print(f"\n📊 System Status:")
                    print(f"LLM Provider: {status['llm_providers']['selected']}")
                    print(f"Active Agents: {status['manager']['total_agents']}")
                    print(f"PDF Files: {status['pdf_knowledge_base']['pdf_count']}")
                    continue
                
                if user_input.lower() in ['test']:
                    print("\n🧪 Testing system components...")
                    test_results = system.test_system()
                    for agent, result in test_results.items():
                        print(f"{agent}: {result}")
                    continue
                
                if user_input.lower() in ['help']:
                    print(system.add_pdfs_info())
                    continue
                
                if not user_input:
                    continue
                
                print("\n🤔 Processing your query...")
                response = system.query(user_input)
                print(f"\n🤖 Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Error in interactive mode: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        logger.error(f"System initialization error: {e}")
        
        # Show available providers for debugging
        providers = get_available_providers()
        print(f"\nAvailable LLM providers: {providers}")
        
        if not any(providers.values()):
            print("\n💡 No LLM providers configured. Please:")
            print("1. Copy config/config_template.py to config/config.py")
            print("2. Add your API keys to config/config.py")
            print("3. Install required packages: pip install -r requirements.txt")


def example_usage():
    """Show example usage of the system"""
    print("🔍 Example Usage of Multi-Agent RAG System")
    print("=" * 50)
    
    try:
        # Initialize system
        system = MultiAgentRAGSystem()
        system.initialize()
        
        # Example queries
        example_queries = [
            "What can you help me with?",
            "What's in my PDF knowledge base?",
            "What's the weather like today?",  # Should use web search
            "Give me a comprehensive analysis of artificial intelligence trends",  # Should use text generation
        ]
        
        for query in example_queries:
            print(f"\n❓ Query: {query}")
            try:
                response = system.query(query)
                print(f"🤖 Response: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
            print("-" * 30)
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "example":
            example_usage()
        elif sys.argv[1] == "test":
            try:
                system = MultiAgentRAGSystem()
                system.initialize()
                results = system.test_system()
                print("🧪 Test Results:")
                for agent, result in results.items():
                    print(f"{agent}: {result}")
            except Exception as e:
                print(f"❌ Test failed: {e}")
        else:
            print("Usage: python main.py [example|test]")
    else:
        # Default to interactive mode
        interactive_mode()
