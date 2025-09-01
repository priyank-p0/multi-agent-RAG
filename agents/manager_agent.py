"""
Manager Agent for Multi-Agent RAG System

This is the central orchestrating agent that coordinates between different specialized agents.
"""

import logging
from typing import Optional, List, Dict, Any

from smolagents import CodeAgent

from utils.llm_providers import get_llm_model
from agents.web_search_agent import create_managed_web_search_agent
from agents.retriever_agent import create_managed_retriever_agent
from agents.text_generation_agent import create_managed_text_generation_agent
from config import CONFIG

logger = logging.getLogger(__name__)


class ManagerAgent:
    """
    Central manager agent that orchestrates multiple specialized agents
    """
    
    def __init__(self, 
                 llm_provider: str = None,
                 enable_web_search: bool = None,
                 enable_retrieval: bool = True,
                 enable_text_generation: bool = None,
                 pdf_directory: str = None,
                 model_kwargs: dict = None):
        """
        Initialize the manager agent
        
        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", "gemini")
            enable_web_search: Enable web search agent
            enable_retrieval: Enable PDF retrieval agent
            enable_text_generation: Enable comprehensive text generation agent
            pdf_directory: Directory containing PDF files
            model_kwargs: Additional arguments for the LLM model
        """
        self.llm_provider = llm_provider or CONFIG["llm"].default_provider
        self.model_kwargs = model_kwargs or {}
        
        # Configuration
        self.enable_web_search = enable_web_search if enable_web_search is not None else CONFIG["agent"].enable_web_search
        self.enable_retrieval = enable_retrieval
        self.enable_text_generation = enable_text_generation if enable_text_generation is not None else CONFIG["agent"].enable_text_generation
        
        # Get LLM model
        try:
            self.model = get_llm_model(self.llm_provider, **self.model_kwargs)
            logger.info(f"Initialized manager agent with {self.llm_provider} provider")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise
        
        # Initialize managed agents
        self.managed_agents = []
        self._initialize_agents(pdf_directory)
        
        # Create the main agent
        self.agent = CodeAgent(
            tools=[],  # Manager agent has no direct tools, only managed agents
            model=self.model,
            managed_agents=self.managed_agents,
            additional_authorized_imports=["time", "datetime", "PIL", "pathlib", "os"],
            max_iterations=CONFIG["agent"].max_iterations,
            verbose=CONFIG["agent"].verbose
        )
        
        logger.info(f"Manager agent initialized with {len(self.managed_agents)} sub-agents")
    
    def _initialize_agents(self, pdf_directory: str = None):
        """Initialize the managed agents based on configuration"""
        
        # Store individual agents for text generation coordination
        self.web_agent = None
        self.retrieval_agent = None
        
        # Web search agent
        if self.enable_web_search:
            try:
                web_agent = create_managed_web_search_agent(
                    llm_provider=self.llm_provider,
                    model_kwargs=self.model_kwargs
                )
                self.managed_agents.append(web_agent)
                self.web_agent = web_agent.agent  # Store for text generation coordination
                logger.info("Web search agent initialized")
            except Exception as e:
                logger.error(f"Failed to initialize web search agent: {e}")
        
        # PDF retrieval agent
        if self.enable_retrieval:
            try:
                retrieval_agent = create_managed_retriever_agent(
                    llm_provider=self.llm_provider,
                    pdf_directory=pdf_directory,
                    model_kwargs=self.model_kwargs
                )
                self.managed_agents.append(retrieval_agent)
                self.retrieval_agent = retrieval_agent.agent  # Store for text generation coordination
                logger.info("PDF retrieval agent initialized")
            except Exception as e:
                logger.error(f"Failed to initialize retrieval agent: {e}")
        
        # Text generation agent (coordinates with web search and retrieval)
        if self.enable_text_generation:
            try:
                text_agent = create_managed_text_generation_agent(
                    llm_provider=self.llm_provider,
                    web_search_agent=self.web_agent,
                    retrieval_agent=self.retrieval_agent,
                    model_kwargs=self.model_kwargs
                )
                self.managed_agents.append(text_agent)
                logger.info("Text generation agent initialized")
            except Exception as e:
                logger.error(f"Failed to initialize text generation agent: {e}")
    
    def run(self, query: str) -> str:
        """
        Process a user query using the appropriate agents
        
        Args:
            query: User query or request
            
        Returns:
            Response from the appropriate agent(s)
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Enhanced system prompt for better agent selection
            system_prompt = f"""
            You are a manager agent coordinating multiple specialized agents:
            
            Available agents:
            {self._get_agent_descriptions()}
            
            Your job is to:
            1. Analyze the user's query
            2. Determine which agent(s) can best help
            3. Call the appropriate agent(s) with the right parameters
            4. Synthesize and present the results
            
            Guidelines for agent selection:
            - Use web_search_agent for current events, news, real-time information, or when local knowledge base doesn't have the answer
            - Use pdf_retriever_agent for information that might be in the local PDF documents
            - Use text_generation_agent for comprehensive research-based responses that combine web search and document retrieval
            - You can use multiple agents if needed (e.g., search web and local docs, then synthesize)
            - Always provide helpful, accurate responses based on the available information
            
            User Query: {query}
            """
            
            result = self.agent.run(system_prompt)
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _get_agent_descriptions(self) -> str:
        """Get descriptions of available agents"""
        descriptions = []
        
        for agent in self.managed_agents:
            name = agent.name
            description = agent.description.strip()
            descriptions.append(f"- {name}: {description}")
        
        return "\n".join(descriptions) if descriptions else "No agents available"
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents
        
        Returns:
            Dictionary with agent status information
        """
        status = {
            "manager": {
                "llm_provider": self.llm_provider,
                "total_agents": len(self.managed_agents),
                "enabled_features": {
                    "web_search": self.enable_web_search,
                    "pdf_retrieval": self.enable_retrieval,
                    "text_generation": self.enable_text_generation
                }
            },
            "agents": []
        }
        
        for agent in self.managed_agents:
            agent_info = {
                "name": agent.name,
                "description": agent.description.strip()[:100] + "..." if len(agent.description.strip()) > 100 else agent.description.strip()
            }
            status["agents"].append(agent_info)
        
        return status
    
    def test_agents(self) -> Dict[str, str]:
        """
        Test all agents with simple queries
        
        Returns:
            Dictionary with test results for each agent
        """
        test_results = {}
        
        for agent in self.managed_agents:
            agent_name = agent.name
            logger.info(f"Testing agent: {agent_name}")
            
            try:
                if "web_search" in agent_name:
                    test_query = "What is the current date?"
                elif "retriever" in agent_name:
                    test_query = "Get information about the PDF knowledge base"
                elif "text_generation" in agent_name:
                    test_query = "Generate a comprehensive response about artificial intelligence"
                else:
                    test_query = "Hello, can you respond?"
                
                # Use the agent directly for testing
                result = agent.agent.run(test_query)
                test_results[agent_name] = f"✓ Success: {result[:100]}..."
                
            except Exception as e:
                test_results[agent_name] = f"✗ Error: {str(e)}"
        
        return test_results


def create_manager_agent(llm_provider: str = None, **kwargs) -> ManagerAgent:
    """
    Factory function to create a manager agent
    
    Args:
        llm_provider: LLM provider to use
        **kwargs: Additional arguments
        
    Returns:
        ManagerAgent instance
    """
    return ManagerAgent(llm_provider=llm_provider, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create manager agent
        manager = create_manager_agent()
        
        # Get status
        status = manager.get_agent_status()
        print("Manager Agent Status:")
        print(f"LLM Provider: {status['manager']['llm_provider']}")
        print(f"Total Agents: {status['manager']['total_agents']}")
        print(f"Enabled Features: {status['manager']['enabled_features']}")
        print("\nAvailable Agents:")
        for agent in status['agents']:
            print(f"- {agent['name']}: {agent['description']}")
        
        # Test a simple query
        print("\nTesting with a simple query...")
        result = manager.run("What can you help me with?")
        print(f"Response: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys in config/config.py")
