import os
import yaml
from typing import List, Dict, Any
from dotenv import load_dotenv
from loguru import logger

from smart_log_analyser.data import LogParser
from smart_log_analyser.embedding.chunker import LogChunker
from smart_log_analyser.embedding.embedder import LogEmbedder
from smart_log_analyser.vector_store.chroma_store import ChromaStore
from smart_log_analyser.analysis.llm_analyzer import LLMAnalyzer
from smart_log_analyser.generator import Generator
class SmartLogAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        load_dotenv()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.parser = LogParser(
            timestamp_format=self.config['log_processing']['timestamp_format']
        )
        self.chunker = LogChunker(
            chunk_size=self.config['chunking']['chunk_size'],
            chunk_overlap=self.config['chunking']['chunk_overlap'],
            separator=self.config['chunking']['separator']
        )
        self.embedder = LogEmbedder(config_path=config_path)
        self.vector_store = ChromaStore(config_path=config_path)
        self.llm_analyzer = LLMAnalyzer(config_path=config_path)
        self.generator = Generator(config_path=config_path)

    def process_logs(
        self,
        log_lines: List[str],
        format_type: str = "text",
        analyze: bool = True
    ) -> Dict[str, Any]:
        try:
            parsed_logs = self.parser.parse_batch(log_lines, format_type)
            logger.info(f"Parsed {len(parsed_logs)} log entries")
            chunks = self.chunker.chunk_logs(parsed_logs)
            logger.info(f"Created {len(chunks)} chunks")
            chunks_with_embeddings = self.embedder.generate_embeddings(chunks)
            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
            success = self.vector_store.add_embeddings(chunks_with_embeddings)
            if success:
                logger.info("Successfully stored embeddings in vector database")
            else:
                logger.error("Failed to store embeddings in vector database")
            analysis_results = {}
            if analyze and success:
                analysis_results["general"] = self.llm_analyzer.analyze_logs(parsed_logs)
                analysis_results["anomalies"] = self.llm_analyzer.detect_anomalies(parsed_logs)
                if analysis_results["anomalies"]["status"] == "success":
                    analysis_results["root_cause"] = self.llm_analyzer.analyze_root_cause(parsed_logs)
            return {
                "status": "success" if success else "error",
                "parsed_logs": len(parsed_logs),
                "chunks": len(chunks),
                "analysis": analysis_results if analyze else None
            }
        except Exception as e:
            logger.error(f"Error processing logs: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def search_logs(
        self,
        query: str,
        n_results: int = 5,
        analyze: bool = True
    ) -> Dict[str, Any]:
        try:
            # Use the query string directly for search
            results = self.vector_store.search(query=query, n_results=n_results)
            analysis_results = {}
            if analyze and results:
                log_entries = [
                    {
                        "content": result["content"],
                        "metadata": result["metadata"]
                    }
                    for result in results
                ]
                analysis_results = self.llm_analyzer.analyze_logs(log_entries)
            return {
                "status": "success",
                "results": results,
                "analysis": analysis_results if analyze else None
            }
        except Exception as e:
            logger.error(f"Error searching logs: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "config": {
                "chunking": self.config['chunking'],
                "embedding": self.config['embedding'],
                "vector_store": self.config['vector_store'],
                "llm": self.config.get('llm', {})
            }
        }

def main():
    analyzer = SmartLogAnalyzer()
    log_lines = [
        "2024-03-20 10:15:30 [INFO] User login successful",
        "2024-03-20 10:15:35 [ERROR] Database connection failed",
        "2024-03-20 10:15:40 [WARNING] High memory usage detected",
        "2024-03-20 10:15:45 [ERROR] Failed to process request",
        "2024-03-20 10:15:50 [INFO] System backup completed"
    ]
    result = analyzer.process_logs(log_lines, analyze=True)
    print("\nProcessing Results:")
    print(f"Status: {result['status']}")
    print(f"Parsed Logs: {result['parsed_logs']}")
    print(f"Chunks: {result['chunks']}")
    if result.get('analysis'):
        print("\nAnalysis Results:")
        for analysis_type, analysis_result in result['analysis'].items():
            print(f"\n{analysis_type.upper()} Analysis:")
            print(analysis_result['result'])
    search_result = analyzer.search_logs("database error", analyze=True)
    print("\nSearch Results:")
    for result in search_result['results']:
        print(f"Found: {result['content']} (Distance: {result['distance']})")
    if search_result.get('analysis'):
        print("\nSearch Analysis:")
        print(search_result['analysis']['result'])

if __name__ == "__main__":
    main() 