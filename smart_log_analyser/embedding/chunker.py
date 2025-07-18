from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[separator, ".", "!", "?", ",", " ", ""],
            length_function=len,
        )

    def chunk_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of parsed logs into smaller segments.
        
        Args:
            logs: List of parsed log entries
            
        Returns:
            List of chunked log entries with metadata
        """
        try:
            # Combine logs into a single text
            combined_text = self._combine_logs(logs)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(combined_text)
            
            # Convert chunks back to structured format
            chunked_logs = []
            for i, chunk in enumerate(chunks):
                chunked_logs.append({
                    "chunk_id": i,
                    "content": chunk,
                    "metadata": {
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks),
                        "chunk_index": i
                    }
                })
            
            return chunked_logs
            
        except Exception as e:
            logger.error(f"Error chunking logs: {str(e)}")
            return []

    def _combine_logs(self, logs: List[Dict[str, Any]]) -> str:
        """
        Combine multiple log entries into a single text string.
        
        Args:
            logs: List of parsed log entries
            
        Returns:
            Combined text string
        """
        combined = []
        for log in logs:
            # Format each log entry
            timestamp = log.get("timestamp", "")
            level = log.get("level", "INFO")
            message = log.get("message", "")
            metadata = log.get("metadata", {})
            
            # Create formatted log entry
            log_entry = f"[{timestamp}] [{level}] {message}"
            if metadata:
                log_entry += f" {metadata}"
            
            combined.append(log_entry)
        
        return "\n".join(combined)

    def chunk_single_log(self, log: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single log entry into smaller segments.
        
        Args:
            log: Single parsed log entry
            
        Returns:
            List of chunked log segments
        """
        return self.chunk_logs([log]) 