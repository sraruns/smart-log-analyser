from typing import List, Dict, Any
import yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger

class LogEmbedder:
    def __init__(
        self,
        config_path: str = "config/config.yaml"
    ):
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        embedding_cfg = config['embedding']
        llm_cfg = config['llm']
        self.model_name = embedding_cfg.get('model_name', 'models/embedding-001')
        self.batch_size = embedding_cfg.get('batch_size', 100)
        self.api_key = llm_cfg.get('api_key')
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key
        )

    def generate_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        try:
            texts = [chunk["content"] for chunk in chunks]
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                embeddings.extend(batch_embeddings)
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            return chunks
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []

    def generate_single_embedding(self, text: str) -> List[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            return []

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [] 