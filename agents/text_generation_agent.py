"""
Text Generation Agent for Multi-Agent RAG System

This agent generates comprehensive text responses by coordinating with web search 
and document retrieval agents to provide well-informed answers.
"""

import logging
from typing import Optional, List, Dict, Any

from smolagents import CodeAgent, Tool

from utils.llm_providers import get_llm_model
from config import CONFIG

logger = logging.getLogger(__name__)


class ComprehensiveTextGenerator(Tool):
    """Tool for generating comprehensive text responses using multiple information sources"""
    
    name = "comprehensive_text_generator"
    description = """
    Generates comprehensive, well-researched text responses by combining information from 
    web search results and local PDF documents. Provides detailed, accurate answers 
    with proper context and citations.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The question or topic to research and generate a comprehensive response for.",
        },
        "include_web_search": {
            "type": "boolean", 
            "description": "Whether to include web search results in the response (default: True)."
        },
        "include_documents": {
            "type": "boolean",
            "description": "Whether to include local PDF document search in the response (default: True)."
        }
    }
    output_type = "string"

    def __init__(self, web_search_agent=None, retrieval_agent=None, **kwargs):
        super().__init__(**kwargs)
        self.web_search_agent = web_search_agent
        self.retrieval_agent = retrieval_agent

    def forward(self, query: str, include_web_search: bool = True, include_documents: bool = True) -> str:
        """
        Generate comprehensive text response using multiple sources
        
        Args:
            query: The question or topic to research
            include_web_search: Whether to search the web
            include_documents: Whether to search local documents
            
        Returns:
            Comprehensive text response with sources
        """
        assert isinstance(query, str), "Query must be a string"
        
        logger.info(f"Generating comprehensive response for: {query}")
        
        # Collect information from different sources
        sources = []
        web_info = ""
        doc_info = ""
        
        # Get web search results if enabled
        if include_web_search and self.web_search_agent:
            try:
                logger.info("Searching web for relevant information...")
                web_info = self.web_search_agent.run(f"Search for comprehensive information about: {query}")
                if web_info and len(web_info.strip()) > 0:
                    sources.append("Web Search")
                    logger.info("Web search completed successfully")
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                web_info = f"Web search unavailable: {str(e)}"
        
        # Get document search results if enabled
        if include_documents and self.retrieval_agent:
            try:
                logger.info("Searching local documents for relevant information...")
                doc_info = self.retrieval_agent.run(f"Search for documents related to: {query}")
                if doc_info and len(doc_info.strip()) > 0:
                    sources.append("Local Documents")
                    logger.info("Document search completed successfully")
            except Exception as e:
                logger.error(f"Document search failed: {e}")
                doc_info = f"Document search unavailable: {str(e)}"
        
        # Generate comprehensive response
        response = self._synthesize_response(query, web_info, doc_info, sources)
        
        logger.info("Comprehensive response generated successfully")
        return response

    def _synthesize_response(self, query: str, web_info: str, doc_info: str, sources: List[str]) -> str:
        """
        Synthesize information from multiple sources into a comprehensive response
        
        Args:
            query: Original query
            web_info: Information from web search
            doc_info: Information from document search
            sources: List of sources used
            
        Returns:
            Synthesized comprehensive response
        """
        
        # Create a structured response
        response_parts = []
        
        # Header
        response_parts.append(f"# Comprehensive Response: {query}")
        response_parts.append("=" * 60)
        
        # Main answer section
        response_parts.append("\n## Summary")
        
        if not web_info and not doc_info:
            response_parts.append("I was unable to gather information from the available sources. Please check your query or try rephrasing it.")
            return "\n".join(response_parts)
        
        # Synthesize the main answer
        main_answer = self._create_main_answer(query, web_info, doc_info)
        response_parts.append(main_answer)
        
        # Add source-specific information
        if web_info and web_info.strip() and "Web Search" in sources:
            response_parts.append("\n## Web Search Results")
            response_parts.append("Based on current online information:")
            response_parts.append(self._clean_and_format_info(web_info))
        
        if doc_info and doc_info.strip() and "Local Documents" in sources:
            response_parts.append("\n## Relevant Documents")
            response_parts.append("From your local document collection:")
            response_parts.append(self._clean_and_format_info(doc_info))
        
        # Footer with sources
        response_parts.append(f"\n## Information Sources")
        if sources:
            response_parts.append(f"This response was compiled using: {', '.join(sources)}")
        else:
            response_parts.append("This response was generated using available system knowledge.")
        
        response_parts.append(f"\n*Response generated for query: \"{query}\"*")
        
        return "\n".join(response_parts)

    def _create_main_answer(self, query: str, web_info: str, doc_info: str) -> str:
        """Create the main synthesized answer"""
        
        # Simple synthesis logic - in a real implementation, you might use an LLM for this
        answer_parts = []
        
        # Extract key information
        if web_info and "Error" not in web_info:
            web_summary = self._extract_key_points(web_info)
            if web_summary:
                answer_parts.append(f"Current information indicates: {web_summary}")
        
        if doc_info and "Error" not in doc_info:
            doc_summary = self._extract_key_points(doc_info)
            if doc_summary:
                answer_parts.append(f"Your documents show: {doc_summary}")
        
        if not answer_parts:
            return f"Based on the available information, I can provide context about {query}. Please see the detailed sections below for comprehensive information from the available sources."
        
        return " ".join(answer_parts)

    def _extract_key_points(self, text: str, max_length: int = 200) -> str:
        """Extract key points from text"""
        if not text or len(text.strip()) < 10:
            return ""
        
        # Simple extraction - take first meaningful sentence or paragraph
        sentences = text.split('.')
        key_info = ""
        
        for sentence in sentences[:3]:  # Look at first 3 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful sentence
                key_info = sentence + "."
                break
        
        if len(key_info) > max_length:
            key_info = key_info[:max_length] + "..."
        
        return key_info

    def _clean_and_format_info(self, info: str) -> str:
        """Clean and format information for display"""
        if not info:
            return "No information available."
        
        # Basic cleaning
        cleaned = info.strip()
        
        # Limit length for readability
        if len(cleaned) > 2000:
            cleaned = cleaned[:2000] + "\n\n[... truncated for readability ...]"
        
        return cleaned


class TextGenerationAgent:
    """Agent for generating comprehensive text responses using multiple information sources"""
    
    def __init__(self, 
                 llm_provider: str = None, 
                 web_search_agent=None,
                 retrieval_agent=None,
                 model_kwargs: dict = None):
        """
        Initialize the text generation agent
        
        Args:
            llm_provider: LLM provider to use ("openai", "anthropic", "gemini")
            web_search_agent: Web search agent instance
            retrieval_agent: Document retrieval agent instance
            model_kwargs: Additional arguments for the LLM model
        """
        self.llm_provider = llm_provider or CONFIG["llm"].default_provider
        self.model_kwargs = model_kwargs or {}
        self.web_search_agent = web_search_agent
        self.retrieval_agent = retrieval_agent
        
        # Get LLM model
        try:
            self.model = get_llm_model(self.llm_provider, **self.model_kwargs)
            logger.info(f"Initialized text generation agent with {self.llm_provider} provider")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise
        
        # Initialize tools
        self.tools = [
            ComprehensiveTextGenerator(
                web_search_agent=self.web_search_agent,
                retrieval_agent=self.retrieval_agent
            )
        ]
        
        # Create the agent (using CodeAgent for better coordination)
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_iterations=CONFIG["agent"].max_iterations,
            verbose=CONFIG["agent"].verbose
        )
        
        logger.info("Text generation agent initialized successfully")
    
    def generate_response(self, query: str, use_web_search: bool = True, use_documents: bool = True) -> str:
        """
        Generate a comprehensive text response
        
        Args:
            query: The question or topic to research
            use_web_search: Whether to include web search
            use_documents: Whether to include document search
            
        Returns:
            Comprehensive text response
        """
        try:
            logger.info(f"Generating comprehensive response for: {query}")
            
            # Create the prompt for the agent
            agent_prompt = f"""
            Generate a comprehensive, well-researched response for the following query: "{query}"
            
            Instructions:
            - Use the comprehensive_text_generator tool to gather information
            - Include web search: {use_web_search}
            - Include local documents: {use_documents}
            - Provide a detailed, accurate, and well-structured response
            - Include proper context and cite sources when available
            
            Query: {query}
            """
            
            result = self.agent.run(agent_prompt)
            logger.info("Text response generated successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error generating text response: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_managed_agent(self):
        """
        Get the agent for use in multi-agent systems
        
        Returns:
            Agent instance with metadata
        """
        # Create a simple wrapper that includes the agent and metadata
        class AgentWrapper:
            def __init__(self, agent, name, description):
                self.agent = agent
                self.name = name
                self.description = description
        
        return AgentWrapper(
            agent=self.agent,
            name="text_generation_agent",
            description="""
            Generates comprehensive text responses by researching topics using both web search 
            and local document retrieval. Provides detailed, well-sourced answers to user questions.
            Use this agent when you need thorough, research-backed text responses.
            Provide your question or topic as an argument.
            """
        )


def create_text_generation_agent(llm_provider: str = None, 
                                web_search_agent=None,
                                retrieval_agent=None,
                                **kwargs) -> TextGenerationAgent:
    """
    Factory function to create a text generation agent
    
    Args:
        llm_provider: LLM provider to use
        web_search_agent: Web search agent instance
        retrieval_agent: Document retrieval agent instance
        **kwargs: Additional arguments
        
    Returns:
        TextGenerationAgent instance
    """
    return TextGenerationAgent(
        llm_provider=llm_provider,
        web_search_agent=web_search_agent,
        retrieval_agent=retrieval_agent,
        **kwargs
    )


def create_managed_text_generation_agent(llm_provider: str = None,
                                        web_search_agent=None,
                                        retrieval_agent=None,
                                        **kwargs):
    """
    Factory function to create a managed text generation agent
    
    Args:
        llm_provider: LLM provider to use
        web_search_agent: Web search agent instance
        retrieval_agent: Document retrieval agent instance
        **kwargs: Additional arguments
        
    Returns:
        Agent wrapper instance
    """
    agent = create_text_generation_agent(
        llm_provider=llm_provider,
        web_search_agent=web_search_agent,
        retrieval_agent=retrieval_agent,
        **kwargs
    )
    return agent.get_managed_agent()


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create text generation agent (without sub-agents for testing)
        text_agent = create_text_generation_agent()
        
        # Test text generation
        result = text_agent.generate_response(
            "What is artificial intelligence and how is it used today?",
            use_web_search=False,  # Disable for testing without sub-agents
            use_documents=False
        )
        print("Text generation result:")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys in the .env file")
