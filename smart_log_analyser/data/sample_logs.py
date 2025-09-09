"""Sample log data for testing and demonstration."""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os


class SampleLogs:
    """Sample log data for testing and demonstration with time-based separation."""
    
   
    @staticmethod
    def get_current_logs() -> List[str]:
        """Get current log entries for analysis (today's logs)."""
        return [
            "2024-01-15 10:30:15 INFO: Application started successfully",
            "2024-01-15 10:30:16 INFO: Database connection established",
            "2024-01-15 10:31:22 WARNING: High memory usage detected: 85% of 8GB used",
            "2024-01-15 10:32:45 ERROR: Failed to process payment request: Invalid card number",
            "2024-01-15 10:33:12 INFO: User login successful for user@example.com",
            "2024-01-15 10:34:28 WARNING: Disk space low: 95% full on /var/log partition",
            "2024-01-15 10:35:44 ERROR: Service unavailable: Downstream service timeout",
            "2024-01-15 10:36:15 INFO: Cache cleared successfully",
            "2024-01-15 10:37:22 CRITICAL: Database connection lost - attempting reconnection",
            "2024-01-15 10:38:01 INFO: User logout successful for user@example.com"
        ]
    
    @staticmethod
    def get_sample_logs() -> List[str]:
        """Get current log entries for backward compatibility."""
        return SampleLogs.get_current_logs()
   
    
    @staticmethod
    def get_logs_from_elasticsearch(
        es_url: Optional[str] = None,
        index_pattern: str = "logs-*",
        size: int = 1000,
        time_range: str = "7d"
    ) -> List[str]:
        """
        Read logs from Elasticsearch.
        
        Args:
            es_url: Elasticsearch URL (defaults to env var ELASTICSEARCH_URL)
            index_pattern: Index pattern to search
            size: Number of logs to retrieve
            time_range: Time range (e.g., "7d", "24h", "1w")
            
        Returns:
            List of formatted log strings
        """
        try:
            from elasticsearch import Elasticsearch
            
            # Get ES URL from parameter or environment variable
            if not es_url:
                es_url = os.getenv("ELASTICSEARCH_URL")
                if not es_url:
                    raise ValueError("Elasticsearch URL not provided and ELASTICSEARCH_URL env var not set")
            
            # Initialize ES client
            es = Elasticsearch([es_url])
            
            # Build query
            query = {
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": f"now-{time_range}"
                        }
                    }
                },
                "sort": [
                    {"@timestamp": {"order": "desc"}}
                ],
                "size": size
            }
            
            # Execute search
            response = es.search(index=index_pattern, body=query)
            
            # Format logs
            logs = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                # Extract common fields
                timestamp = source.get('@timestamp', '')
                level = source.get('level', source.get('log.level', 'INFO'))
                message = source.get('message', source.get('log.message', ''))
                
                # Format as log string
                if timestamp and message:
                    # Convert ISO timestamp to readable format
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_time = timestamp
                    
                    log_string = f"{formatted_time} {level}: {message}"
                    logs.append(log_string)
            
            return logs
            
        except ImportError:
            raise ImportError("elasticsearch package not installed. Run: pip install elasticsearch")
        except Exception as e:
            raise Exception(f"Failed to read from Elasticsearch: {str(e)}")
