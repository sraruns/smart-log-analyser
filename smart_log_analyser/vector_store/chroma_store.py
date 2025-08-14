from typing import List, Dict, Any, Optional
import os
import hashlib
import json
from datetime import datetime
import yaml
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from .base import VectorStoreBase

# Load environment variables from .env file
load_dotenv()

class ChromaStore(VectorStoreBase):
    def __init__(
        self,
        config_path: str = "config/config.yaml"
    ):
        super().__init__(config_path)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        vs_cfg = config['vector_store']
        emb_cfg = config['embedding']
        
        self.collection_name = vs_cfg.get('collection_name', 'log_embeddings')
        self.persist_directory = vs_cfg.get('persist_directory', './data/vector_store')
        self.current_version = vs_cfg.get('version', '1.0.0')
        
        # Get Google API key from environment variables
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=emb_cfg.get('model_name', 'models/embedding-001'),
            google_api_key=google_api_key
        )
        
        # Initialize with metadata for version tracking
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_metadata={"version": self.current_version}
        )
        
        # Store the config hash to detect changes
        self.config_hash = self._generate_config_hash(config)
        
    def _generate_config_hash(self, config: Dict) -> str:
        """Generate a hash of relevant config settings"""
        # Only include relevant config sections
        config_subset = {
            'chunking': config.get('chunking', {}),
            'embedding': config.get('embedding', {}),
            'search': config.get('search', {})
        }
        return hashlib.md5(json.dumps(config_subset, sort_keys=True).encode()).hexdigest()
        
    def needs_refresh(self, config_path: str) -> bool:
        """Check if the vector store needs refresh based on config changes"""
        try:
            with open(config_path, 'r') as f:
                current_config = yaml.safe_load(f)
            
            current_hash = self._generate_config_hash(current_config)
            
            # Get stored version and hash if available
            collection = self.chroma.get_collection()
            stored_hash = collection.metadata.get('config_hash', '')
            
            return current_hash != stored_hash
            
        except Exception as e:
            logger.warning(f"Error checking if refresh needed: {str(e)}")
            return True  # Default to refresh on error
            
    def update_metadata(self, config_path: str):
        """Update stored metadata with current config"""
        try:
            with open(config_path, 'r') as f:
                current_config = yaml.safe_load(f)
                
            collection = self.chroma.get_collection()
            collection.modify(
                metadata={
                    **collection.metadata,
                    'version': self.current_version,
                    'config_hash': self._generate_config_hash(current_config),
                    'updated_at': datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            
    def delete_old_versions(self, keep_versions: int = 2):
        """Clean up old versions of chunks"""
        try:
            collection = self.chroma.get_collection()
            
            # Get all chunks with their versions
            results = collection.get(include=['metadatas'])
            metadatas = results.get('metadatas', [])
            
            # Find chunks older than N versions
            to_delete = []
            for idx, meta in enumerate(metadatas):
                chunk_version = meta.get('version', '1.0.0')
                if self._version_compare(chunk_version, self.current_version) < 0:
                    to_delete.append(idx)
            
            if to_delete:
                # Delete old chunks
                ids_to_delete = [results['ids'][i] for i in to_delete]
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} old chunks")
                
        except Exception as e:
            logger.error(f"Error deleting old versions: {str(e)}")
            
    def _version_compare(self, v1: str, v2: str) -> int:
        """Compare two version strings"""
        def normalize(v):
            return [int(x) for x in v.split('.')]
            
        v1_parts = normalize(v1)
        v2_parts = normalize(v2)
        
        # Pad with zeros if needed
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for i in range(max_len):
            if v1_parts[i] > v2_parts[i]:
                return 1
            elif v1_parts[i] < v2_parts[i]:
                return -1
        return 0
        
    def add_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        try:
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            
            # Add version to metadata
            for meta in metadatas:
                meta['version'] = self.current_version
                meta['created_at'] = datetime.utcnow().isoformat()
            
            self.chroma.add_texts(documents, metadatas=metadatas, ids=ids)
            self.chroma.persist()
            return True
        except Exception as e:
            logger.error(f"Error adding embeddings to vector store: {str(e)}")
            return False

    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            # Load search configuration
            with open("config/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            search_cfg = config.get('search', {})
            
            similarity_threshold = search_cfg.get('similarity_threshold', 0.4)
            max_results = search_cfg.get('max_results', n_results)
            include_combined = search_cfg.get('include_combined_chunks', False)
            
            # Preprocess query for better matching
            processed_query = self._preprocess_query(query)
            
            results = self.chroma.similarity_search_with_score(processed_query, k=max_results * 2)  # Get more to filter
            formatted_results = []
            
            for doc, score in results:
                # Filter by similarity threshold
                if score <= similarity_threshold:
                    # Filter combined chunks if not wanted
                    chunk_type = doc.metadata.get('chunk_type', 'individual_log')
                    if not include_combined and chunk_type == 'combined_context':
                        continue
                        
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "distance": score,
                        "relevance_score": 1.0 - score  # Higher is better
                    })
            
            # Sort by distance (lower is better) and limit results
            formatted_results.sort(key=lambda x: x['distance'])
            return formatted_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess search query to match log format better.
        
        Args:
            query: Original search query
            
        Returns:
            Processed query optimized for log search
        """
        # Convert to uppercase for log levels if it looks like a level
        log_levels = ['ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'FATAL']
        words = query.split()
        
        processed_words = []
        for word in words:
            upper_word = word.upper()
            if upper_word in log_levels:
                processed_words.append(f"{upper_word}:")
            else:
                processed_words.append(word)
        
        return " ".join(processed_words)

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection = self.chroma.get_collection()
            return {
                'count': collection.count(),
                'version': collection.metadata.get('version'),
                'last_updated': collection.metadata.get('updated_at'),
                'config_hash': collection.metadata.get('config_hash')
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

    def delete_collection(self) -> bool:
        try:
            self.chroma.delete_collection()
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False