# Multi-Agent RAG System 🤖🤝🤖

A multi-agent Retrieval-Augmented Generation (RAG) system that combines multiple specialized agents to handle diverse tasks including web search, document retrieval from local PDFs, and comprehensive text generation with research capabilities.

## Features

- **Multi-Agent Architecture**: Coordinated agents for different specializations
- **PDF Knowledge Base**: Process and search through your local PDF documents
- **Web Search**: Real-time information retrieval from the internet
- **Comprehensive Text Generation**: Research-backed responses combining web search and document retrieval
- **Multiple LLM Providers**: Support for OpenAI, Anthropic (Claude), and Google Gemini
- **Intelligent Orchestration**: Central manager agent routes queries to appropriate specialists

## Architecture

```
Manager Agent (Orchestrator)
├── Web Search Agent (DuckDuckGo + Web Scraping)
├── PDF Retriever Agent (Local Document Search)
└── Text Generation Agent (Research-backed Response Generation)
    ├── Coordinates with Web Search Agent
    └── Coordinates with PDF Retriever Agent
```

## Quick Start

### 1. Installation

```bash
# Clone or create the project directory
cd multi-agent-RAG

# Create virtual environment (if not already created)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the environment template
cp env.template .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Example `.env` configuration:
```bash
# Add at least one API key
OPENAI_API_KEY=sk-your-actual-openai-key-here
ANTHROPIC_API_KEY=your-actual-anthropic-key-here
GOOGLE_API_KEY=your-actual-gemini-key-here

# Optional: Set your preferred provider
DEFAULT_LLM_PROVIDER=openai

# Optional: Customize other settings
CHUNK_SIZE=512
MAX_RETRIEVAL_DOCS=7
ENABLE_WEB_SEARCH=true
ENABLE_TEXT_GENERATION=true
```

### 3. Add PDF Documents (Optional)

```bash
# Add your PDF files to the knowledge base
cp your_documents/*.pdf data/pdfs/
```

### 4. Run the System

```bash
# Interactive mode
python main.py

# Test the system
python main.py test

# See example usage
python main.py example
```

## Usage Examples

### Interactive Mode
```bash
python main.py
```

Example interactions:
```
Ask me anything: What's the latest news about AI agents?
Response: [Uses web search agent to find current information]

Ask me anything: Search my documents for information about machine learning
Response: [Uses PDF retriever agent to search local documents]

Ask me anything: Give me a comprehensive analysis of quantum computing
Response: [Uses text generation agent to provide research-backed analysis]
```

### Programmatic Usage
```python
from main import MultiAgentRAGSystem

# Initialize the system
system = MultiAgentRAGSystem(llm_provider="openai")
system.initialize()

# Query the system
response = system.query("Explain quantum computing")
print(response)

# Get system status
status = system.get_system_status()
print(status)
```

## Configuration Options

The system uses environment variables (`.env` file) for configuration:

```bash
# API Keys (at least one required)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here

# LLM Settings
DEFAULT_LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4-turbo-preview
ANTHROPIC_MODEL=claude-3-sonnet-20240229
GEMINI_MODEL=gemini-pro

# RAG Settings
PDF_DIRECTORY=data/pdfs
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_RETRIEVAL_DOCS=7

# Agent Settings
MAX_ITERATIONS=4
VERBOSE=true
ENABLE_WEB_SEARCH=true
ENABLE_TEXT_GENERATION=true
```