from typing import List, Dict, Any
import os
from langchain_openai import OpenAIEmbeddings
from loguru import logger

class LogEmbedder:
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 100
    ):
        """
        Initialize the LogEmbedder with specified parameters.
        
        Args:
            model_name: Name of the OpenAI embedding model to use
            batch_size: Number of texts to embed in a single batch
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of log chunks.
        
        Args:
            chunks: List of chunked log entries
            
        Returns:
            List of chunks with their embeddings
        """
        try:
            # Extract texts from chunks
            texts = [chunk["content"] for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                embeddings.extend(batch_embeddings)
            
            # Combine chunks with their embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            return []

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [] 