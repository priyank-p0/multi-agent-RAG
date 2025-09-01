"""
Retriever Agent for Multi-Agent RAG System

This agent handles document retrieval from the local PDF knowledge base.
"""

import logging
from typing import Optional, List
from pathlib import Path

from smolagents import Tool, ToolCallingAgent
from langchain_core.vectorstores import VectorStore

from utils.llm_providers import get_llm_model
from utils.pdf_processor import PDFProcessor
from config import CONFIG

logger = logging.getLogger(__name__)


class PDFRetrieverTool(Tool):
    """Tool for retrieving documents from PDF knowledge base"""
    
    name = "pdf_retriever"
    description = """
    Retrieves relevant documents from the local PDF knowledge base using semantic similarity.
    The knowledge base contains PDF documents that have been processed and embedded.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to search for. This should be semantically close to your target documents. Use descriptive terms rather than questions.",
        }
    }
    output_type = "string"

    def __init__(self, pdf_processor: PDFProcessor, **kwargs):
        super().__init__(**kwargs)
        self.pdf_processor = pdf_processor
        self.max_docs = CONFIG["rag"].max_retrieval_docs

    def forward(self, query: str) -> str:
        """
        Retrieve relevant documents based on query
        
        Args:
            query: Search query
            
        Returns:
            Formatted string with retrieved documents
        """
        assert isinstance(query, str), "Your search query must be a string"

        try:
            logger.info(f"Searching PDF knowledge base for: {query}")
            
            # Search documents
            docs = self.pdf_processor.search_documents(query, k=self.max_docs)
            
            if not docs:
                return f"No relevant documents found for query: {query}"
            
            # Format results
            result = f"\nRetrieved {len(docs)} documents from PDF knowledge base:\n"
            result += "=" * 50 + "\n"
            
            for i, doc in enumerate(docs):
                filename = doc.metadata.get('filename', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                
                result += f"\n--- Document {i+1} (from {filename}) ---\n"
                result += doc.page_content
                result += f"\n[Source: {source}]\n"
                result += "-" * 30 + "\n"
            
            logger.info(f"Retrieved {len(docs)} documents successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            logger.error(error_msg)
            return error_msg


class PDFKnowledgeBaseInfoTool(Tool):
    """Tool for getting information about the PDF knowledge base"""
    
    name = "pdf_kb_info"
    description = """
    Provides information about the PDF knowledge base, including the number of documents,
    available files, and status of the vector store.
    """
    inputs = {}
    output_type = "string"

    def __init__(self, pdf_processor: PDFProcessor, **kwargs):
        super().__init__(**kwargs)
        self.pdf_processor = pdf_processor

    def forward(self) -> str:
        """
        Get information about the PDF knowledge base
        
        Returns:
            Information about the knowledge base
        """
        try:
            pdf_count = self.pdf_processor.get_pdf_count()
            pdf_files = self.pdf_processor.list_pdf_files()
            
            info = f"PDF Knowledge Base Information:\n"
            info += f"- Number of PDF files: {pdf_count}\n"
            info += f"- PDF directory: {self.pdf_processor.pdf_directory}\n"
            
            if pdf_files:
                info += f"- Available files:\n"
                for filename in pdf_files:
                    info += f"  • {filename}\n"
            else:
                info += f"- No PDF files found in directory\n"
                info += f"  Please add PDF files to {self.pdf_processor.pdf_directory}\n"
            
            # Check vector store status
            vector_store_path = CONFIG["rag"].vector_store_path
            if Path(vector_store_path).exists():
                info += f"- Vector store: Available at {vector_store_path}\n"
            else:
                info += f"- Vector store: Not created yet (will be created on first search)\n"
            
            return info
            
        except Exception as e:
            error_msg = f"Error getting knowledge base info: {str(e)}"
            logger.error(error_msg)
            return error_msg


class RetrieverAgent:
    """Agent for handling document retrieval from PDF knowledge base"""
    
    def __init__(self, 
                 llm_provider: str = None, 
                 pdf_directory: str = None,
                 model_kwargs: dict = None):
        """
        Initialize the retriever agent
        
        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", "gemini")
            pdf_directory: Directory containing PDF files
            model_kwargs: Additional arguments for the LLM model
        """
        self.llm_provider = llm_provider or CONFIG["llm"].default_provider
        self.model_kwargs = model_kwargs or {}
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(pdf_directory=pdf_directory)
        
        # Get LLM model
        try:
            self.model = get_llm_model(self.llm_provider, **self.model_kwargs)
            logger.info(f"Initialized retriever agent with {self.llm_provider} provider")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise
        
        # Initialize tools
        self.tools = [
            PDFRetrieverTool(self.pdf_processor),
            PDFKnowledgeBaseInfoTool(self.pdf_processor)
        ]
        
        # Create the agent
        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=self.model,
            max_iterations=CONFIG["agent"].max_iterations,
            verbose=CONFIG["agent"].verbose
        )
        
        logger.info("Retriever agent initialized successfully")
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            
        Returns:
            Retrieved documents as formatted string
        """
        try:
            logger.info(f"Retrieving documents for: {query}")
            result = self.agent.run(f"Search for documents related to: {query}")
            return result
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return f"Error during document retrieval: {str(e)}"
    
    def get_knowledge_base_info(self) -> str:
        """
        Get information about the knowledge base
        
        Returns:
            Information about the PDF knowledge base
        """
        try:
            result = self.agent.run("Get information about the PDF knowledge base")
            return result
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {e}")
            return f"Error getting knowledge base info: {str(e)}"
    
    def rebuild_vector_store(self) -> str:
        """
        Rebuild the vector store from PDF files
        
        Returns:
            Status message
        """
        try:
            logger.info("Rebuilding vector store from PDF files...")
            self.pdf_processor.get_or_create_vector_store(force_recreate=True)
            return "Vector store rebuilt successfully from PDF files"
        except Exception as e:
            error_msg = f"Error rebuilding vector store: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_managed_agent(self):
        """
        Get the agent wrapper for use in multi-agent systems
        
        Returns:
            Agent wrapper instance
        """
        # Create a simple wrapper that includes the agent and metadata
        class AgentWrapper:
            def __init__(self, agent, name, description):
                self.agent = agent
                self.name = name
                self.description = description
        
        return AgentWrapper(
            agent=self.agent,
            name="pdf_retriever_agent",
            description="""
            Retrieves relevant documents from the local PDF knowledge base.
            The knowledge base contains PDF documents that have been processed and embedded for semantic search.
            Use this agent when you need information from the local document collection.
            Provide your search query as an argument.
            """
        )


def create_retriever_agent(llm_provider: str = None, 
                          pdf_directory: str = None,
                          **kwargs) -> RetrieverAgent:
    """
    Factory function to create a retriever agent
    
    Args:
        llm_provider: LLM provider to use
        pdf_directory: Directory containing PDF files
        **kwargs: Additional arguments
        
    Returns:
        RetrieverAgent instance
    """
    return RetrieverAgent(
        llm_provider=llm_provider,
        pdf_directory=pdf_directory,
        **kwargs
    )


def create_managed_retriever_agent(llm_provider: str = None,
                                  pdf_directory: str = None,
                                  **kwargs):
    """
    Factory function to create a managed retriever agent
    
    Args:
        llm_provider: LLM provider to use
        pdf_directory: Directory containing PDF files
        **kwargs: Additional arguments
        
    Returns:
        Agent wrapper instance
    """
    agent = create_retriever_agent(
        llm_provider=llm_provider,
        pdf_directory=pdf_directory,
        **kwargs
    )
    return agent.get_managed_agent()


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create retriever agent
        retriever_agent = create_retriever_agent()
        
        # Get knowledge base info
        info = retriever_agent.get_knowledge_base_info()
        print("Knowledge Base Info:")
        print(info)
        
        # Test retrieval (if PDFs are available)
        pdf_count = retriever_agent.pdf_processor.get_pdf_count()
        if pdf_count > 0:
            result = retriever_agent.retrieve("machine learning")
            print("\nRetrieval Result:")
            print(result)
        else:
            print("\nNo PDF files found. Add some PDF files to test retrieval.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys in config/config.py")
