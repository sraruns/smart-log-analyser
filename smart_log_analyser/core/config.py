from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from loguru import logger


class ConfigManager:
    """Centralized configuration management for Smart Log Analyzer."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration."""
        return self._config
    
    @property
    def log_processing(self) -> Dict[str, Any]:
        """Get log processing configuration."""
        return self._config.get('log_processing', {})
    
    @property
    def chunking(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return self._config.get('chunking', {})
    
    @property
    def embedding(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self._config.get('embedding', {})
    
    @property
    def vector_store(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self._config.get('vector_store', {})
    
    @property
    def search(self) -> Dict[str, Any]:
        """Get search configuration."""
        return self._config.get('search', {})
    
    @property
    def llm(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get('llm', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set a configuration value by key."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, backup: bool = True):
        """Save configuration to file."""
        try:
            if backup and self.config_path.exists():
                backup_path = self.config_path.with_suffix('.yaml.bak')
                self.config_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            raise
    
    def validate_config(self) -> bool:
        """Validate configuration completeness."""
        required_sections = ['log_processing', 'chunking', 'embedding', 'vector_store', 'search', 'llm']
        
        for section in required_sections:
            if section not in self._config:
                logger.error(f"Missing required config section: {section}")
                return False
        
        # Validate specific required fields
        required_fields = {
            'log_processing.timestamp_format': str,
            'chunking.chunk_size': int,
            'embedding.batch_size': int,
            'vector_store.collection_name': str,
            'llm.model_name': str
        }
        
        for field, expected_type in required_fields.items():
            value = self.get(field)
            if value is None:
                logger.error(f"Missing required config field: {field}")
                return False
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for {field}: expected {expected_type.__name__}, got {type(value).__name__}")
                return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_vector_store_type(self) -> str:
        """Get the configured vector store type."""
        return self.get('vector_store.type', 'chroma')
    
    def get_embedding_model_type(self) -> str:
        """Get the configured embedding model type."""
        return self.get('embedding.model_type', 'auto')
    
    def update_vector_store_config(self, store_type: str, **kwargs):
        """Update vector store configuration."""
        self.set('vector_store.type', store_type)
        
        for key, value in kwargs.items():
            self.set(f'vector_store.{key}', value)
    
    def update_embedding_config(self, model_type: str, **kwargs):
        """Update embedding configuration."""
        self.set('embedding.model_type', model_type)
        
        for key, value in kwargs.items():
            self.set(f'embedding.{key}', value)
