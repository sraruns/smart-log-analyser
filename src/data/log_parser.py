import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Union
from loguru import logger

class LogParser:
    def __init__(self, timestamp_format: str = "%Y-%m-%d %H:%M:%S"):
        self.timestamp_format = timestamp_format
        self.patterns = {
            "json": self._parse_json_log,
            "text": self._parse_text_log,
            "syslog": self._parse_syslog
        }

    def parse(self, log_line: str, format_type: str = "text") -> Dict:
        """
        Parse a single log line based on the specified format.
        
        Args:
            log_line: The log line to parse
            format_type: The format of the log (json, text, or syslog)
            
        Returns:
            Dict containing parsed log information
        """
        try:
            if format_type not in self.patterns:
                raise ValueError(f"Unsupported log format: {format_type}")
            
            return self.patterns[format_type](log_line)
        except Exception as e:
            logger.error(f"Error parsing log line: {str(e)}")
            return {"raw": log_line, "error": str(e)}

    def _parse_json_log(self, log_line: str) -> Dict:
        """Parse a JSON formatted log line."""
        try:
            log_data = json.loads(log_line)
            return {
                "timestamp": log_data.get("timestamp"),
                "level": log_data.get("level", "INFO"),
                "message": log_data.get("message", ""),
                "metadata": {k: v for k, v in log_data.items() 
                           if k not in ["timestamp", "level", "message"]}
            }
        except json.JSONDecodeError:
            return {"raw": log_line, "error": "Invalid JSON format"}

    def _parse_text_log(self, log_line: str) -> Dict:
        """Parse a text formatted log line."""
        # Basic pattern for timestamp, level, and message
        pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[?(\w+)\]? (.*)"
        match = re.match(pattern, log_line)
        
        if match:
            timestamp, level, message = match.groups()
            return {
                "timestamp": timestamp,
                "level": level,
                "message": message.strip(),
                "metadata": {}
            }
        return {"raw": log_line, "error": "Could not parse text log format"}

    def _parse_syslog(self, log_line: str) -> Dict:
        """Parse a syslog formatted log line."""
        # Syslog pattern: timestamp hostname program[pid]: message
        pattern = r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}) (\S+) (\S+)(?:\[(\d+)\])?: (.*)"
        match = re.match(pattern, log_line)
        
        if match:
            timestamp, hostname, program, pid, message = match.groups()
            return {
                "timestamp": timestamp,
                "hostname": hostname,
                "program": program,
                "pid": pid,
                "message": message.strip(),
                "metadata": {
                    "hostname": hostname,
                    "program": program,
                    "pid": pid
                }
            }
        return {"raw": log_line, "error": "Could not parse syslog format"}

    def parse_batch(self, log_lines: List[str], format_type: str = "text") -> List[Dict]:
        """
        Parse multiple log lines.
        
        Args:
            log_lines: List of log lines to parse
            format_type: The format of the logs
            
        Returns:
            List of parsed log entries
        """
        return [self.parse(line, format_type) for line in log_lines] 