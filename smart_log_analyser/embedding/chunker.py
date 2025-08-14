from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from smart_log_analyser.data.log_parser import LogParser
import yaml
class LogChunker:
    def __init__(
        
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n"
    ):
        """
        Initialize the LogChunker with specified parameters.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: Character(s) to use for splitting text
        """
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            chunk_size = config['chunking']['chunk_size']
            chunk_overlap = config['chunking']['chunk_overlap']
            separator = config['chunking']['separator']
 
        logger.info(f"Chunk size: {chunk_size} Chunk overlap: {chunk_overlap} Separator: {separator}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[separator, ".", "!", "?", ",", " ", ""],
            length_function=len,
        )

    def chunk_logs(self, parsed_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk parsed logs into manageable segments for embedding.
        
        Args:
            parsed_logs: List of parsed log entries
            
        Returns:
            List of chunked log segments with metadata
        """
        if not parsed_logs:
            return []
        
        

        try:
            with open("config/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            search_cfg = config.get('search', {})
            include_combined = search_cfg.get('include_combined_chunks', False)
        except:
            include_combined = False
        
        # For log analysis, we'll use individual log entries as chunks
        # This provides better precision for similarity search
        chunks = []
        
        for i, log in enumerate(parsed_logs):
            # Extract semantic content for better embeddings
            parser = LogParser()
            semantic_content = parser.extract_semantic_content(log)
            
            chunk = {
                "chunk_id": f"log_{i}",
                "content": semantic_content,  # Use semantic content instead of raw log
                "metadata": {
                    "original_content": self._format_log_entry(log),
                    "log_level": log.get("level", "INFO"),
                    "timestamp": log.get("timestamp", ""),
                    "chunk_index": i,
                    "chunk_type": "individual_log"
                }
            }
            chunks.append(chunk)
        
        # Only create combined chunks if explicitly enabled in config
        if include_combined and len(chunks) > 3:  # Create combined chunks for better context
            combined_content = self._combine_logs_for_context(parsed_logs[:10])  # Limit to prevent huge chunks
            if combined_content:
                chunks.append({
                    "chunk_id": f"combined_context_{len(chunks)}",
                    "content": combined_content,
                    "metadata": {
                        "chunk_type": "combined_context",
                        "log_count": min(len(parsed_logs), 10),
                        "chunk_index": len(chunks)
                    }
                })
        
        logger.info(f"Created {len(chunks)} chunks from {len(parsed_logs)} logs")
        return chunks
    
    def _combine_logs_for_context(self, parsed_logs: List[Dict[str, Any]]) -> str:
        """
        Combine multiple logs into a single chunk for broader context.
        
        Args:
            parsed_logs: List of parsed log entries
            
        Returns:
            Combined semantic content
        """
        parser = LogParser()
        
        semantic_contents = []
        for log in parsed_logs:
            semantic_content = parser.extract_semantic_content(log)
            semantic_contents.append(semantic_content)
        
        return "\n".join(semantic_contents)

    def _format_log_entry(self, log: Dict[str, Any]) -> str:
        """
        Format a single log entry into a string.
        
        Args:
            log: Parsed log entry
            
        Returns:
            Formatted log entry string
        """
        timestamp = log.get("timestamp", "")
        level = log.get("level", "INFO")
        message = log.get("message", "")
        metadata = log.get("metadata", {})
        
        log_entry = f"[{timestamp}] [{level}] {message}"
        if metadata:
            log_entry += f" {metadata}"
        
        return log_entry

    def chunk_single_log(self, log: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single log entry into smaller segments.
        
        Args:
            log: Single parsed log entry
            
        Returns:
            List of chunked log segments
        """
        return self.chunk_logs([log])