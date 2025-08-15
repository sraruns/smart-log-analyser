# Smart Log Analyzer

An intelligent log analysis system that uses LLMs, LangChain, and vector databases to provide deep insights into log data with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Advanced Log Processing**: Intelligent parsing and preprocessing of log data
- **Smart Chunking**: Context-aware chunking of log data for optimal analysis
- **Multiple Embedding Options**: Support for OpenAI embeddings and sentence transformers
- **Vector Database Storage**: Flexible storage with ChromaDB and Qdrant support
- **RAG-Enhanced Analysis**: Retrieval-augmented generation for context-aware log analysis
- **Semantic Search**: Advanced search capabilities across log data
- **Pattern Recognition**: Intelligent detection of log patterns and anomalies
- **LLM Integration**: Support for Google Gemini and other LLM providers
- **Comprehensive Query Interface**: Rich interface for log analysis and insights

## Project Structure

```
smart-log-analyser/
├── smart_log_analyser/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── smart_log_analyzer.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── log_parser.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── embedder_simple.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chroma_store.py
│   │   └── qdrant_store.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── llm_analyzer.py
│   │   ├── prompt_manager.py
│   │   └── rag_chain.py
│   └── main.py
├── config/
│   └── config.yaml
├── data/
│   └── vector_store/
├── tests/
│   ├── __init__.py
│   ├── test_embedding.py
│   ├── test_llm_analyzer.py
│   └── test_search.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10.18 (exact version required)
- OpenAI API key (for OpenAI embeddings)
- Google API key (for Gemini LLM integration)

## Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd smart-log-analyser
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the root directory with your API keys:
```bash
# OpenAI API key (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Google API key (for Gemini LLM)
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Basic Usage

1. **Run the main application:**
```bash
python smart_log_analyser/main.py
```

2. **Run tests:**
```bash
pytest tests/
```

### Advanced Features

The system supports:
- **RAG Chain Analysis**: Enhanced log analysis using retrieval-augmented generation
- **Multiple Vector Stores**: Choose between ChromaDB and Qdrant
- **Flexible Embedding Models**: OpenAI embeddings or sentence transformers
- **Context-Aware Processing**: Intelligent chunking and analysis based on log context

## Configuration

The `config/config.yaml` file contains all configurable parameters:
- Chunking settings and strategies
- Embedding model parameters
- Vector database configuration (ChromaDB/Qdrant)
- LLM provider settings
- Analysis thresholds and parameters

## Dependencies

Key dependencies include:
- **LangChain**: Core LLM framework integration
- **Google Generative AI**: Gemini LLM support
- **ChromaDB & Qdrant**: Vector database options
- **Sentence Transformers**: Alternative embedding models
- **PyTorch**: Machine learning backend
- **Loguru**: Enhanced logging capabilities

## Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
- **Core**: Main analyzer logic and configuration
- **Data**: Log parsing and preprocessing
- **Embedding**: Text chunking and embedding generation
- **Vector Store**: Database abstraction and implementations
- **Analysis**: LLM integration and RAG chain implementation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License

## Support

For issues and questions, please check the existing issues or create a new one in the repository.