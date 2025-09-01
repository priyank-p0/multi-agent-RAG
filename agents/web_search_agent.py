"""
Web Search Agent for Multi-Agent RAG System

This agent handles web searches using DuckDuckGo and can visit web pages to extract content.
"""

import logging
from typing import Optional

from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool

from utils.llm_providers import get_llm_model
from config import CONFIG

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """Web search agent with DuckDuckGo search and webpage visiting capabilities"""
    
    def __init__(self, llm_provider: str = None, model_kwargs: dict = None):
        """
        Initialize the web search agent
        
        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", "gemini")
            model_kwargs: Additional arguments for the LLM model
        """
        self.llm_provider = llm_provider or CONFIG["llm"].default_provider
        self.model_kwargs = model_kwargs or {}
        
        # Get LLM model
        try:
            self.model = get_llm_model(self.llm_provider, **self.model_kwargs)
            logger.info(f"Initialized web search agent with {self.llm_provider} provider")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise
        
        # Initialize tools
        self.tools = [
            DuckDuckGoSearchTool(),
            VisitWebpageTool()
        ]
        
        # Create the agent
        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=self.model,
            max_iterations=CONFIG["agent"].max_iterations,
            verbose=CONFIG["agent"].verbose
        )
        
        logger.info("Web search agent initialized successfully")
    
    def search(self, query: str) -> str:
        """
        Perform a web search
        
        Args:
            query: Search query
            
        Returns:
            Search results as formatted string
        """
        try:
            logger.info(f"Performing web search for: {query}")
            result = self.agent.run(f"Search the web for: {query}")
            return result
        except Exception as e:
            logger.error(f"Error during web search: {e}")
            return f"Error performing web search: {str(e)}"
    
    def search_and_visit(self, query: str, max_pages: int = 3) -> str:
        """
        Search the web and visit top results to get detailed content
        
        Args:
            query: Search query
            max_pages: Maximum number of pages to visit
            
        Returns:
            Detailed content from search results
        """
        try:
            logger.info(f"Performing detailed web search for: {query}")
            detailed_prompt = f"""
            Search the web for "{query}" and then visit the top {max_pages} most relevant pages to get detailed information.
            Provide a comprehensive summary of the information found.
            """
            result = self.agent.run(detailed_prompt)
            return result
        except Exception as e:
            logger.error(f"Error during detailed web search: {e}")
            return f"Error performing detailed web search: {str(e)}"
    
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
            name="web_search_agent",
            description="""
            Performs web searches using DuckDuckGo and can visit web pages to extract content.
            Use this agent when you need current information from the internet.
            Provide your search query as an argument.
            """
        )


def create_web_search_agent(llm_provider: str = None, **kwargs) -> WebSearchAgent:
    """
    Factory function to create a web search agent
    
    Args:
        llm_provider: LLM provider to use
        **kwargs: Additional arguments
        
    Returns:
        WebSearchAgent instance
    """
    return WebSearchAgent(llm_provider=llm_provider, **kwargs)


def create_managed_web_search_agent(llm_provider: str = None, **kwargs):
    """
    Factory function to create a managed web search agent
    
    Args:
        llm_provider: LLM provider to use
        **kwargs: Additional arguments
        
    Returns:
        Agent wrapper instance
    """
    agent = create_web_search_agent(llm_provider=llm_provider, **kwargs)
    return agent.get_managed_agent()


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create web search agent
        web_agent = create_web_search_agent()
        
        # Test search
        result = web_agent.search("What is the latest news about AI agents?")
        print("Search Result:")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys in config/config.py")
