from typing import List, Dict, Any, Optional
import os
import hashlib
import json
import uuid
from datetime import datetime
import yaml
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False

from .base import VectorStoreBase


class QdrantStore(VectorStoreBase):
    """Qdrant vector store implementation with improved persistence and versioning."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path)
        
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vs_cfg = config['vector_store']
        self.collection_name = vs_cfg.get('collection_name', 'log_embeddings')
        self.current_version = vs_cfg.get('version', '1.0.0')
        
        # Qdrant configuration
        qdrant_cfg = vs_cfg.get('qdrant', {})
        self.host = qdrant_cfg.get('host', 'localhost')
        self.port = qdrant_cfg.get('port', 6333)
        self.url = qdrant_cfg.get('url', None)
        self.api_key = qdrant_cfg.get('api_key', None)
        
        # Initialize Qdrant client
        if self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(host=self.host, port=self.port)
        
        # Vector dimensions (will be set when first embedding is added)
        self.vector_size = qdrant_cfg.get('vector_size', 384)  # Default for sentence-transformers
        
        # Store config hash for change detection
        self.config_hash = self._generate_config_hash(config)
        
        # Initialize collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _generate_config_hash(self, config: Dict) -> str:
        """Generate a hash of relevant config settings."""
        config_subset = {
            'chunking': config.get('chunking', {}),
            'embedding': config.get('embedding', {}),
            'search': config.get('search', {})
        }
        return hashlib.md5(json.dumps(config_subset, sort_keys=True).encode()).hexdigest()
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
                
                # Set initial metadata
                self._set_collection_metadata({
                    'version': self.current_version,
                    'config_hash': self.config_hash,
                    'created_at': datetime.utcnow().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    def _set_collection_metadata(self, metadata: Dict[str, Any]):
        """Set collection metadata using payload."""
        try:
            # Generate a consistent UUID for metadata based on collection name
            metadata_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"metadata_{self.collection_name}"))
            
            # Store metadata as a special point with UUID
            metadata_point = PointStruct(
                id=metadata_uuid,
                vector=[0.0] * self.vector_size,  # Dummy vector
                payload={
                    "is_metadata": True,
                    **metadata
                }
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[metadata_point]
            )
        except Exception as e:
            logger.error(f"Error setting collection metadata: {str(e)}")
            raise

    def _get_collection_metadata(self) -> Dict[str, Any]:
        """Get collection metadata."""
        try:
            # Generate the same UUID as used in _set_collection_metadata
            metadata_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"metadata_{self.collection_name}"))
            
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[metadata_uuid],
                with_payload=True,
                with_vectors=False
            )
            
            if result and hasattr(result[0], 'payload'):
                payload = result[0].payload
                # Remove internal flags
                if payload and isinstance(payload, dict):
                    payload.pop('is_metadata', None)
                    return payload
            return {}
        except Exception as e:
            logger.warning(f"Error getting collection metadata: {str(e)}")
            return {}
    
    def needs_refresh(self, config_path: str) -> bool:
        """Check if vector store needs refresh based on config changes."""
        try:
            with open(config_path, 'r') as f:
                current_config = yaml.safe_load(f)
            
            current_hash = self._generate_config_hash(current_config)
            metadata = self._get_collection_metadata()
            stored_hash = metadata.get('config_hash', '')
            
            return current_hash != stored_hash
            
        except Exception as e:
            logger.warning(f"Error checking if refresh needed: {str(e)}")
            return True  # Default to refresh on error
    
    def update_metadata(self, config_path: str) -> bool:
        """Update collection metadata."""
        try:
            with open(config_path, 'r') as f:
                current_config = yaml.safe_load(f)
            
            metadata = self._get_collection_metadata()
            metadata.update({
                'version': self.current_version,
                'config_hash': self._generate_config_hash(current_config),
                'updated_at': datetime.utcnow().isoformat()
            })
            
            self._set_collection_metadata(metadata)
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return False
    
    def add_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add embeddings to Qdrant."""
        try:
            points = []
            for chunk in chunks:
                if 'embedding' not in chunk:
                    logger.warning(f"Chunk {chunk.get('chunk_id', 'unknown')} missing embedding")
                    continue
                
                # Ensure vector size matches collection configuration
                embedding = chunk['embedding']
                if len(embedding) != self.vector_size:
                    logger.warning(f"Embedding size mismatch: {len(embedding)} vs {self.vector_size}")
                    continue
                
                # Prepare metadata
                metadata = chunk.get('metadata', {})
                metadata.update({
                    'version': self.current_version,
                    'created_at': datetime.utcnow().isoformat(),
                    'chunk_id': chunk.get('chunk_id', str(uuid.uuid4()))
                })
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'content': chunk['content'],
                        **metadata
                    }
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Added {len(points)} embeddings to Qdrant")
                return True
            else:
                logger.warning("No valid embeddings to add")
                return False
                
        except Exception as e:
            logger.error(f"Error adding embeddings to Qdrant: {str(e)}")
            return False
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant."""
        try:
            # Load search configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            search_cfg = config.get('search', {})
            
            similarity_threshold = search_cfg.get('similarity_threshold', 0.7)
            include_combined = search_cfg.get('include_combined_chunks', False)
            
            # Note: query embedding should be generated by the caller
            # This is a placeholder - in practice, you'd pass the query embedding
            logger.warning("Search requires query embedding to be generated externally")
            return []
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            return []
    
    def search_with_embedding(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search with pre-computed query embedding."""
        try:
            # Load search configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            search_cfg = config.get('search', {})
            
            similarity_threshold = search_cfg.get('similarity_threshold', 0.7)
            include_combined = search_cfg.get('include_combined_chunks', False)
            
            # Build filter for excluding metadata points
            must_not_conditions = [
                FieldCondition(key="is_metadata", match=MatchValue(value=True))
            ]
            
            # Filter combined chunks if not wanted
            if not include_combined:
                must_not_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchValue(value="combined_context"))
                )
            
            search_filter = Filter(must_not=must_not_conditions) if must_not_conditions else None
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=n_results * 2,  # Get more to filter by threshold
                query_filter=search_filter,
                with_payload=True,
                score_threshold=1.0 - similarity_threshold  # Qdrant uses similarity, not distance
            )
            
            # Format results
            formatted_results = []
            for result in results:
                if result.score >= (1.0 - similarity_threshold):
                    formatted_results.append({
                        "content": result.payload.get('content', ''),
                        "metadata": {k: v for k, v in result.payload.items() if k != 'content'},
                        "distance": 1.0 - result.score,  # Convert back to distance
                        "relevance_score": result.score
                    })
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching Qdrant with embedding: {str(e)}")
            return []
    
    def delete_old_versions(self, keep_versions: int = 2) -> bool:
        """Clean up old versions of data."""
        try:
            # Get all points with version information
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your data size
                with_payload=True
            )
            
            points = scroll_result[0]
            
            # Group by chunk_id and sort by version
            version_groups = {}
            for point in points:
                payload = point.payload
                if payload.get('is_metadata'):
                    continue  # Skip metadata points
                
                chunk_id = payload.get('chunk_id')
                version = payload.get('version', '0.0.0')
                
                if chunk_id:
                    if chunk_id not in version_groups:
                        version_groups[chunk_id] = []
                    version_groups[chunk_id].append((point.id, version))
            
            # Delete old versions
            points_to_delete = []
            for chunk_id, versions in version_groups.items():
                # Sort by version (newest first)
                versions.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only the specified number of versions
                if len(versions) > keep_versions:
                    old_versions = versions[keep_versions:]
                    points_to_delete.extend([point_id for point_id, _ in old_versions])
            
            if points_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=points_to_delete
                )
                logger.info(f"Deleted {len(points_to_delete)} old version points")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting old versions: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            metadata = self._get_collection_metadata()
            
            return {
                'count': info.points_count,
                'version': metadata.get('version'),
                'last_updated': metadata.get('updated_at'),
                'config_hash': metadata.get('config_hash'),
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance.value
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
