from typing import List, Dict, Any, Union, Optional
import os
from loguru import logger

from .config import ConfigManager
from ..data.log_parser import LogParser
from ..embedding.chunker import LogChunker
from ..embedding.embedder_simple import SimpleLogEmbedder
from ..vector_store.base import VectorStoreBase
from ..vector_store.chroma_store import ChromaStore
from ..vector_store.qdrant_store import QdrantStore
from ..analysis.llm_analyzer import LLMAnalyzer


class SmartLogAnalyzer:
    """
    Refactored Smart Log Analyzer with modular architecture.
    
    Features:
    - Automatic vector store selection (Chroma/Qdrant)
    - Automatic embedding model selection (Google/SentenceTransformers)
    - Improved versioning and cleanup
    - Better error handling and logging
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the SmartLogAnalyzer with configuration."""
        
        # Initialize configuration manager
        self.config_manager = ConfigManager(config_path)
        
        # Validate configuration
        if not self.config_manager.validate_config():
            raise ValueError("Invalid configuration. Please check your config file.")
        
        # Initialize components
        self._initialize_components()
        
        # Setup vector store with versioning
        self._setup_vector_store()
        
        logger.info("SmartLogAnalyzer initialized successfully")
    
    def _initialize_components(self):
        """Initialize all analyzer components."""
        try:
            # Initialize log parser
            self.log_parser = LogParser(
                timestamp_format=self.config_manager.log_processing.get('timestamp_format')
            )
            
            # Initialize chunker
            self.log_chunker = LogChunker(self.config_manager.config_path)
            
            # Initialize embedder (simplified version to avoid property conflicts)
            self.log_embedder = SimpleLogEmbedder(self.config_manager.config_path)
            
            # Initialize vector store
            self.vector_store = self._create_vector_store()
            
            # Initialize LLM analyzer
            self.llm_analyzer = LLMAnalyzer(self.config_manager.config_path)
            
            logger.info(f"Components initialized:")
            logger.info(f"  - Embedding model: {self.log_embedder.model_name} ({self.log_embedder.dimension}D)")
            logger.info(f"  - Vector store: {type(self.vector_store).__name__}")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def _create_vector_store(self) -> VectorStoreBase:
        """Create appropriate vector store based on configuration."""
        store_type = self.config_manager.get_vector_store_type()
        
        if store_type == 'qdrant':
            try:
                return QdrantStore(self.config_manager.config_path)
            except ImportError:
                logger.warning("Qdrant not available, falling back to Chroma")
                return ChromaStore(self.config_manager.config_path)
        
        elif store_type == 'chroma':
            return ChromaStore(self.config_manager.config_path)
        
        else:
            # Auto-select: prefer Qdrant if available
            try:
                logger.info("Auto-selecting vector store: trying Qdrant first")
                return QdrantStore(self.config_manager.config_path)
            except ImportError:
                logger.info("Qdrant not available, using Chroma")
                return ChromaStore(self.config_manager.config_path)
    
    def _setup_vector_store(self):
        """Setup vector store with version management."""
        try:
            # Check if we need to refresh the vector store
            if self.vector_store.needs_refresh(self.config_manager.config_path):
                logger.info("Configuration changed, refreshing vector store...")
                self.clear_vector_store()
                self.vector_store.update_metadata(self.config_manager.config_path)
            
            # Clean up old versions
            self.vector_store.delete_old_versions(keep_versions=2)
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            # Continue without failing - vector store might still work
    
    def process_logs(
        self, 
        log_input: Union[str, List[str]],
        format_type: str = "text",
        analyze: bool = False
    ) -> Dict[str, Any]:
        """
        Process logs with enhanced pipeline.
        
        Args:
            log_input: Either a file path (str) or a list of log lines
            format_type: The format of the logs (default: "text")
            analyze: Whether to perform LLM analysis (default: False)
            
        Returns:
            Dict containing processing results and analysis if requested
        """
        try:
            # Handle both file path and list of log lines
            log_lines = self._prepare_log_lines(log_input)
            
            if not log_lines:
                return {"status": "error", "message": "No log lines to process"}
            
            # Parse logs
            parsed_logs = self.log_parser.parse_batch(log_lines, format_type)
            
            # Extract semantic content for better embeddings
            processed_logs = self._enhance_logs_for_embedding(parsed_logs)
            
            # Chunk logs
            chunks = self.log_chunker.chunk_logs(processed_logs)
            
            # Generate embeddings
            embedded_chunks = self.log_embedder.generate_embeddings(chunks)
            
            # Add to vector store
            success = self.vector_store.add_embeddings(embedded_chunks)
            
            if not success:
                logger.warning("Failed to add some embeddings to vector store")
            
            # Optional: Analyze logs for anomalies
            analysis = {}
            if analyze:
                analysis = self._perform_llm_analysis(parsed_logs)
            
            return {
                "status": "success",
                "processed_logs": len(processed_logs),
                "chunks_created": len(chunks),
                "embeddings_added": len(embedded_chunks),
                "analysis": analysis,
                "vector_store_stats": self.vector_store.get_collection_stats(),
                "embedding_model": self.log_embedder.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Error processing logs: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _prepare_log_lines(self, log_input: Union[str, List[str]]) -> List[str]:
        """Prepare log lines from input (file or list)."""
        if isinstance(log_input, str):
            # Read from file
            with open(log_input, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        elif isinstance(log_input, list):
            # Use provided list of log lines
            return [line.strip() for line in log_input if line and str(line).strip()]
        else:
            raise ValueError("log_input must be either a file path (str) or a list of log lines")
    
    def _enhance_logs_for_embedding(self, parsed_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance logs with semantic content for better embeddings."""
        processed_logs = []
        for log in parsed_logs:
            if not log.get('content'):
                log['content'] = self.log_parser.extract_semantic_content(log)
            processed_logs.append(log)
        return processed_logs
    
    def _perform_llm_analysis(self, parsed_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform LLM analysis on logs."""
        analysis = {}
        try:
            analysis["anomalies"] = self.llm_analyzer.detect_anomalies(parsed_logs)
            if analysis["anomalies"].get("status") == "success":
                analysis["root_cause"] = self.llm_analyzer.analyze_root_cause(parsed_logs)
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            analysis["error"] = str(e)
        
        return analysis
    
    def search_logs(
        self,
        query: str,
        n_results: int = 5,
        analyze: bool = False
    ) -> Dict[str, Any]:
        """
        Search logs with enhanced retrieval.
        
        Args:
            query: Search query
            n_results: Number of results to return
            analyze: Whether to perform LLM analysis on results
            
        Returns:
            Dict containing search results and optional analysis
        """
        try:
            # Generate query embedding
            query_embedding = self.log_embedder.generate_single_embedding(query)
            
            if not query_embedding:
                return {"status": "error", "message": "Failed to generate query embedding"}
            
            # Search with embedding (for Qdrant) or fallback to text search
            if hasattr(self.vector_store, 'search_with_embedding'):
                results = self.vector_store.search_with_embedding(query_embedding, n_results)
            else:
                # Fallback for stores that don't support embedding search
                results = self.vector_store.search(query, n_results)
            
            # Optional: Add LLM analysis
            analysis = {}
            if analyze and results:
                context = "\n".join([r['content'] for r in results])
                analysis = self._analyze_query_results(query, context)
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "analysis": analysis,
                "vector_store_stats": self.vector_store.get_collection_stats()
            }
            
        except Exception as e:
            logger.error(f"Error searching logs: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_query_results(self, query: str, context: str) -> Dict[str, Any]:
        """Analyze search results with LLM."""
        try:
            # This could be enhanced with specific query analysis
            return {
                "summary": f"Found relevant logs for query: {query}",
                "context_length": len(context),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error analyzing query results: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive analyzer statistics."""
        try:
            return {
                "vector_store": self.vector_store.get_collection_stats(),
                "embedding_model": self.log_embedder.get_model_info(),
                "config": {
                    "chunking": self.config_manager.chunking,
                    "embedding": self.config_manager.embedding,
                    "vector_store": self.config_manager.vector_store,
                    "search": self.config_manager.search
                },
                "components": {
                    "vector_store_type": type(self.vector_store).__name__,
                    "embedding_dimension": self.log_embedder.dimension,
                    "chunk_size": self.config_manager.chunking.get('chunk_size'),
                    "batch_size": self.config_manager.embedding.get('batch_size')
                }
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def clear_vector_store(self) -> bool:
        """
        Clear the vector store and remove all stored embeddings.
        Use this when changing chunking logic or when you need fresh data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = self.vector_store.delete_collection()
            if success:
                logger.info("Vector store cleared successfully")
                # Reinitialize the vector store
                self.vector_store = self._create_vector_store()
                return True
            else:
                logger.error("Failed to clear vector store")
                return False
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def migrate_to_qdrant(self) -> bool:
        """
        Migrate from Chroma to Qdrant vector store.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if isinstance(self.vector_store, QdrantStore):
                logger.info("Already using Qdrant")
                return True
            
            logger.info("Starting migration from Chroma to Qdrant...")
            
            # Update configuration
            self.config_manager.update_vector_store_config('qdrant')
            
            # Create new Qdrant store
            new_store = QdrantStore(self.config_manager.config_path)
            
            # TODO: Implement data migration if needed
            # For now, we'll just switch to the new store
            self.vector_store = new_store
            
            logger.info("Migration to Qdrant completed")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating to Qdrant: {str(e)}")
            return False
    
    def get_embedding_comparison(self) -> Dict[str, Any]:
        """Get comparison of available embedding models."""
        try:
            return {
                "current_model": self.log_embedder.get_model_info(),
                "available_models": ["Google Embeddings", "Sentence Transformers"],
                "recommendation": "Current model was auto-selected as best available"
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding comparison: {str(e)}")
            return {"status": "error", "message": str(e)}
