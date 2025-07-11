from typing import List, Dict, Any
import os
import google.generativeai as genai
from loguru import logger

class LLMAnalyzer:
    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        top_p: float = 0.8,
        top_k: int = 40
    ):
        """
        Initialize the LLM Analyzer with Gemini.
        
        Args:
            model_name: Name of the Gemini model to use
            temperature: Temperature for response generation
            max_output_tokens: Maximum number of tokens in the response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        
        # Define analysis prompts
        self.prompts = {
            "log_analysis": """You are an expert log analyzer. Analyze the following log entries and provide insights:
            1. Identify patterns and anomalies
            2. Suggest potential issues or concerns
            3. Provide recommendations for improvement
            4. Categorize the severity of issues
            
            Log entries:
            {logs}
            
            Provide your analysis in a structured format.""",
            
            "anomaly_detection": """You are an expert in log anomaly detection. Analyze the following log entries and:
            1. Identify any unusual patterns or anomalies
            2. Explain why they are considered anomalies
            3. Suggest potential causes
            4. Rate the severity (Low/Medium/High)
            
            Log entries:
            {logs}
            
            Provide your analysis in a structured format.""",
            
            "root_cause": """You are an expert in root cause analysis. Given the following log entries:
            1. Identify the root cause of any issues
            2. Explain the sequence of events
            3. Suggest preventive measures
            4. Provide actionable recommendations
            
            Log entries:
            {logs}
            
            Provide your analysis in a structured format."""
        }

    def analyze_logs(
        self,
        logs: List[Dict[str, Any]],
        analysis_type: str = "log_analysis"
    ) -> Dict[str, Any]:
        """
        Analyze logs using Gemini.
        
        Args:
            logs: List of log entries to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if analysis_type not in self.prompts:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # Format logs for analysis
            formatted_logs = self._format_logs_for_analysis(logs)
            
            # Get the appropriate prompt
            prompt = self.prompts[analysis_type].format(logs=formatted_logs)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return {
                "analysis_type": analysis_type,
                "result": response.text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logs: {str(e)}")
            return {
                "analysis_type": analysis_type,
                "error": str(e),
                "status": "error"
            }

    def detect_anomalies(
        self,
        logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect anomalies in logs.
        
        Args:
            logs: List of log entries to analyze
            
        Returns:
            Dictionary containing anomaly detection results
        """
        return self.analyze_logs(logs, "anomaly_detection")

    def analyze_root_cause(
        self,
        logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze root cause of issues in logs.
        
        Args:
            logs: List of log entries to analyze
            
        Returns:
            Dictionary containing root cause analysis results
        """
        return self.analyze_logs(logs, "root_cause")

    def _format_logs_for_analysis(self, logs: List[Dict[str, Any]]) -> str:
        """
        Format logs for analysis.
        
        Args:
            logs: List of log entries
            
        Returns:
            Formatted string of logs
        """
        formatted_logs = []
        for log in logs:
            timestamp = log.get("timestamp", "N/A")
            level = log.get("level", "INFO")
            message = log.get("message", log.get("content", ""))
            formatted_logs.append(f"[{timestamp}] [{level}] {message}")
        
        return "\n".join(formatted_logs)