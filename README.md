# Smart Log Analyzer

An intelligent log analysis system that uses LLMs, LangChain, and vector databases to provide deep insights into log data.

## Features

- Log parsing and preprocessing
- Intelligent chunking of log data
- Embedding generation using OpenAI
- Vector database storage with ChromaDB
- Semantic search capabilities
- Log pattern analysis and anomaly detection
- Query interface for log analysis

## Project Structure

```
smart-log-analyzer/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── log_parser.py
│   │   └── preprocessor.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── chroma_store.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── pattern_analyzer.py
│   │   └── anomaly_detector.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_log_parser.py
│   ├── test_chunker.py
│   └── test_embedder.py
├── config/
│   └── config.yaml
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
```

4. Update the `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the main application:
```bash
python src/main.py
```

2. For development, run tests:
```bash
pytest tests/
```

## Configuration

The `config/config.yaml` file contains all configurable parameters:
- Chunking settings
- Embedding model parameters
- Vector database configuration
- Analysis thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License