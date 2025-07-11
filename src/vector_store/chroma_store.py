from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from loguru import logger

class ChromaStore:
    def __init__(
        self,
        collection_name: str = "log_embeddings",
        persist_directory: str = "./data/vector_store"
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection to store embeddings
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Add embeddings to the vector store.
        
        Args:
            chunks: List of chunks with their embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for ChromaDB
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            embeddings = [chunk["embedding"] for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to vector store: {str(e)}")
            return False

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar log entries using an embedding.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of similar log entries
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

    def delete_collection(self) -> bool:
        """
        Delete the current collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {} 