from typing import List, Dict, Any
import yaml
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger

class ChromaStore:
    def __init__(
        self,
        config_path: str = "config/config.yaml"
    ):
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        vs_cfg = config['vector_store']
        emb_cfg = config['embedding']
        llm_cfg = config['llm']
        self.collection_name = vs_cfg.get('collection_name', 'log_embeddings')
        self.persist_directory = vs_cfg.get('persist_directory', './data/vector_store')
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=emb_cfg.get('model_name', 'models/embedding-001')
        )
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )

    def add_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        try:
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
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
            results = self.chroma.similarity_search_with_score(query, k=n_results)
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": score
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            return {"collection_name": self.collection_name, "persist_directory": self.persist_directory}
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