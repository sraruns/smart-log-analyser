from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStoreBase(ABC):
    """Abstract base class for vector store implementations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.current_version = "1.0.0"
    
    @abstractmethod
    def add_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add embeddings to the vector store.
        
        Args:
            chunks: List of chunks with embeddings and metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results with content, metadata, and scores
        """
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the entire collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dict containing collection statistics
        """
        pass
    
    @abstractmethod
    def needs_refresh(self, config_path: str) -> bool:
        """Check if vector store needs refresh based on config changes.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            bool: True if refresh needed, False otherwise
        """
        pass
    
    @abstractmethod
    def update_metadata(self, config_path: str) -> bool:
        """Update collection metadata.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_old_versions(self, keep_versions: int = 2) -> bool:
        """Clean up old versions of data.
        
        Args:
            keep_versions: Number of versions to keep
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
