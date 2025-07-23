from typing import List, Dict, Any, Optional
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from loguru import logger

class LLMAnalyzer:
    def __init__(
        self,
        config_path: str = "config/config.yaml"
    ):
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        llm_cfg = config['llm']
        self.model_name = llm_cfg['model_name']  #Required
        self.temperature = llm_cfg.get('temperature', 0.1)
        self.max_output_tokens = llm_cfg.get('max_output_tokens', 2048)
        self.top_p = llm_cfg.get('top_p', 0.8)
        self.top_k = llm_cfg.get('top_k', 40)
        
        # Convert safety settings list to dictionary format if needed
        self.safety_settings = self._process_safety_settings(
            llm_cfg.get('safety_settings', [])
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k
        )
        self.prompts = {
            "log_analysis": PromptTemplate(
                input_variables=["logs"],
                template="""You are an expert log analyzer. Analyze the following log entries and provide insights:\n1. Identify patterns and anomalies\n2. Suggest potential issues or concerns\n3. Provide recommendations for improvement\n4. Categorize the severity of issues\n\nLog entries:\n{logs}\n\nProvide your analysis in a structured format."""
            ),
            "anomaly_detection": PromptTemplate(
                input_variables=["logs"],
                template="""You are an expert in log anomaly detection. Analyze the following log entries and:\n1. Identify any unusual patterns or anomalies\n2. Explain why they are considered anomalies\n3. Suggest potential causes\n4. Rate the severity (Low/Medium/High)\n\nLog entries:\n{logs}\n\nProvide your analysis in a structured format."""
            ),
            "root_cause": PromptTemplate(
                input_variables=["logs"],
                template="""You are an expert in root cause analysis. Given the following log entries:\n1. Identify the root cause of any issues\n2. Explain the sequence of events\n3. Suggest preventive measures\n4. Provide actionable recommendations\n\nLog entries:\n{logs}\n\nProvide your analysis in a structured format."""
            )
        }

    def _process_safety_settings(self, safety_settings: list) -> dict:
        """Convert safety settings from list format to dictionary format.
        
        Args:
            safety_settings: List of safety setting dictionaries
            
        Returns:
            Dictionary of safety settings in the format expected by ChatGoogleGenerativeAI
        """
        if not safety_settings or not isinstance(safety_settings, list):
            # Default safety settings if none provided
            return {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
            
        # Convert list of {category: ..., threshold: ...} to {category: threshold}
        return {item['category']: item['threshold'] for item in safety_settings}

    def analyze_logs(
        self,
        logs: List[Dict[str, Any]],
        analysis_type: str = "log_analysis"
    ) -> Dict[str, Any]:
        try:
            if analysis_type not in self.prompts:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            formatted_logs = self._format_logs_for_analysis(logs)
            prompt = self.prompts[analysis_type].format(logs=formatted_logs)
            response = self.llm.invoke(prompt)
            return {
                "analysis_type": analysis_type,
                "result": response.content,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error analyzing logs: {str(e)}")
            return {
                "analysis_type": analysis_type,
                "error": str(e),
                "status": "error"
            }

    def detect_anomalies(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.analyze_logs(logs, "anomaly_detection")

    def analyze_root_cause(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.analyze_logs(logs, "root_cause")

    def _format_logs_for_analysis(self, logs: List[Dict[str, Any]]) -> str:
        formatted_logs = []
        for log in logs:
            timestamp = log.get("timestamp", "N/A")
            level = log.get("level", "INFO")
            message = log.get("message", log.get("content", ""))
            formatted_logs.append(f"[{timestamp}] [{level}] {message}")
        return "\n".join(formatted_logs)