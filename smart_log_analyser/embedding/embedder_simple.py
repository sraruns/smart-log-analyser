from typing import List, Dict, Any, Optional
import os
import yaml
from loguru import logger

# Try to import embedding models
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google embeddings not available")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available")


class SimpleLogEmbedder:
    """Simplified log embedder that works reliably."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        embedding_cfg = config['embedding']
        self.batch_size = embedding_cfg.get('batch_size', 100)
        
        # Initialize embedding model
        self.embeddings = None
        self._model_name = None
        self._dimension = None
        
        self._initialize_embeddings(embedding_cfg)
        
        logger.info(f"Initialized simple embedder with {self._model_name} ({self._dimension}D)")

    def _initialize_embeddings(self, embedding_cfg: Dict[str, Any]):
        """Initialize the best available embedding model."""
        
        model_type = embedding_cfg.get('model_type', 'auto')
        
        if model_type == 'google' and GOOGLE_AVAILABLE:
            self._init_google_embeddings(embedding_cfg)
        elif model_type == 'sentence_transformers' and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_sentence_transformers(embedding_cfg)
        elif model_type == 'auto':
            # Auto-select: prefer sentence transformers for cost efficiency
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._init_sentence_transformers(embedding_cfg)
            elif GOOGLE_AVAILABLE:
                self._init_google_embeddings(embedding_cfg)
            else:
                raise ImportError("No embedding models available")
        else:
            raise ValueError(f"Requested model type '{model_type}' not available")
    
    def _init_google_embeddings(self, embedding_cfg: Dict[str, Any]):
        """Initialize Google embeddings."""
        model_name = embedding_cfg.get('model_name', 'models/embedding-001')
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )
        self._model_name = model_name
        self._dimension = 768  # Google embedding dimension
        self._model_type = 'google'
    
    def _init_sentence_transformers(self, embedding_cfg: Dict[str, Any]):
        """Initialize Sentence Transformers."""
        st_config = embedding_cfg.get('sentence_transformers', {})
        model_name = st_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading sentence transformer model on {device}")
        
        # Load model
        self.embeddings = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._dimension = self.embeddings.get_sentence_embedding_dimension()
        self._model_type = 'sentence_transformers'

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks with batching optimization."""
        try:
            texts = [chunk["content"] for chunk in chunks]
            
            # Process in batches for efficiency
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                if self._model_type == 'google':
                    batch_embeddings = self.embeddings.embed_documents(batch_texts)
                else:  # sentence_transformers
                    batch_embeddings = self.embeddings.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
                    batch_embeddings = batch_embeddings.tolist()
                
                embeddings.extend(batch_embeddings)
                
                if len(texts) > self.batch_size:
                    logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            logger.info(f"Generated {len(embeddings)} embeddings using {self._model_name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []

    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            if self._model_type == 'google':
                return self.embeddings.embed_query(text)
            else:  # sentence_transformers
                embedding = self.embeddings.encode([text], convert_to_tensor=False, show_progress_bar=False)
                return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            return []

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            if self._model_type == 'google':
                return self.embeddings.embed_documents(texts)
            else:  # sentence_transformers
                embeddings = self.embeddings.encode(texts, convert_to_tensor=False, show_progress_bar=False)
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return []
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self._model_name
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            'model_name': self._model_name,
            'dimension': self._dimension,
            'type': self._model_type
        }
