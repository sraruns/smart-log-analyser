import os
import yaml
from typing import List, Dict, Any
from dotenv import load_dotenv
from loguru import logger

from .data import LogParser
from .embedding import LogChunker, LogEmbedder
from .vector_store import ChromaStore
from .analysis import LLMAnalyzer

class SmartLogAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Smart Log Analyzer.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.parser = LogParser(
            timestamp_format=self.config['log_processing']['timestamp_format']
        )
        
        self.chunker = LogChunker(
            chunk_size=self.config['chunking']['chunk_size'],
            chunk_overlap=self.config['chunking']['chunk_overlap'],
            separator=self.config['chunking']['separator']
        )
        
        self.embedder = LogEmbedder(
            model_name=self.config['embedding']['model_name'],
            batch_size=self.config['embedding']['batch_size']
        )
        
        self.vector_store = ChromaStore(
            collection_name=self.config['vector_store']['collection_name'],
            persist_directory=self.config['vector_store']['persist_directory']
        )
        
        # Initialize LLM analyzer
        self.llm_analyzer = LLMAnalyzer(
            model_name=self.config.get('llm', {}).get('model_name', 'gpt-3.5-turbo'),
            temperature=self.config.get('llm', {}).get('temperature', 0.1)
        )

    def process_logs(
        self,
        log_lines: List[str],
        format_type: str = "text",
        analyze: bool = True
    ) -> Dict[str, Any]:
        """
        Process a list of log lines through the entire pipeline.
        
        Args:
            log_lines: List of log lines to process
            format_type: Format of the logs (json, text, or syslog)
            analyze: Whether to perform LLM analysis
            
        Returns:
            Dictionary containing processing results and analysis
        """
        try:
            # Parse logs
            parsed_logs = self.parser.parse_batch(log_lines, format_type)
            logger.info(f"Parsed {len(parsed_logs)} log entries")
            
            # Chunk logs
            chunks = self.chunker.chunk_logs(parsed_logs)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            chunks_with_embeddings = self.embedder.generate_embeddings(chunks)
            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
            
            # Store in vector database
            success = self.vector_store.add_embeddings(chunks_with_embeddings)
            if success:
                logger.info("Successfully stored embeddings in vector database")
            else:
                logger.error("Failed to store embeddings in vector database")
            
            # Perform LLM analysis if requested
            analysis_results = {}
            if analyze and success:
                # General log analysis
                analysis_results["general"] = self.llm_analyzer.analyze_logs(parsed_logs)
                
                # Anomaly detection
                analysis_results["anomalies"] = self.llm_analyzer.detect_anomalies(parsed_logs)
                
                # Root cause analysis if anomalies are found
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
        """
        Search for similar log entries using a text query.
        
        Args:
            query: Text query to search for
            n_results: Number of results to return
            analyze: Whether to perform LLM analysis on results
            
        Returns:
            Dictionary containing search results and analysis
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedder.generate_single_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results
            )
            
            # Perform LLM analysis if requested
            analysis_results = {}
            if analyze and results:
                # Convert search results to log format
                log_entries = [
                    {
                        "content": result["content"],
                        "metadata": result["metadata"]
                    }
                    for result in results
                ]
                
                # Analyze search results
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
        """
        Get statistics about the current state of the analyzer.
        
        Returns:
            Dictionary containing statistics
        """
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
    # Initialize analyzer
    analyzer = SmartLogAnalyzer()
    
    # Example usage
    log_lines = [
        "2024-03-20 10:15:30 [INFO] User login successful",
        "2024-03-20 10:15:35 [ERROR] Database connection failed",
        "2024-03-20 10:15:40 [WARNING] High memory usage detected",
        "2024-03-20 10:15:45 [ERROR] Failed to process request",
        "2024-03-20 10:15:50 [INFO] System backup completed"
    ]
    
    # Process logs with analysis
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
    
    # Search for similar logs
    search_result = analyzer.search_logs("database error", analyze=True)
    print("\nSearch Results:")
    for result in search_result['results']:
        print(f"Found: {result['content']} (Distance: {result['distance']})")
    
    if search_result.get('analysis'):
        print("\nSearch Analysis:")
        print(search_result['analysis']['result'])

if __name__ == "__main__":
    main() 