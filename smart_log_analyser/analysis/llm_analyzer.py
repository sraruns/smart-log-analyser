from typing import List, Dict, Any, Optional
import yaml
import os
from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from .prompt_manager import PromptManager




class LLMAnalyzer:
    """
    Concise LLM Analyzer for log analysis with key insights.
    
    Features:
    - Quick anomaly detection
    - Root cause identification
    - Concise log summaries
    - Key metrics extraction
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize LLM analyzer with configuration."""
        self.config_path = config_path
        self._load_config()
        self._initialize_llm()
        self._setup_prompts()
        
        logger.info(f"LLM Analyzer initialized with model: {self.model_name}")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            llm_config = config.get('llm', {})
            
            # Model configuration
            self.model_name = llm_config.get('model_name', 'gemini-pro')
            self.temperature = llm_config.get('temperature', 0.1)
            self.max_tokens = llm_config.get('max_tokens', 1000)
            self.top_p = llm_config.get('top_p', 0.8)
            self.top_k = llm_config.get('top_k', 40)
            
            # Fix safety settings - convert to proper format or disable
            self.safety_settings = None  # Disable safety settings to avoid validation errors
            
            logger.debug(f"LLM config loaded: {self.model_name}, temp={self.temperature}")
            
        except Exception as e:
            logger.error(f"Error loading LLM config: {str(e)}")
            # Set defaults
            self.model_name = 'gemini-pro'
            self.temperature = 0.1
            self.max_tokens = 1000
            self.top_p = 0.8
            self.top_k = 40
            self.safety_settings = None
    
    def _initialize_llm(self):
        """Initialize the LLM model."""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            # Initialize LLM with configuration (without safety_settings to avoid validation errors)
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k
                # Removed safety_settings to fix validation error
            )
            
            logger.debug("LLM model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def _setup_prompts(self):
        """Setup prompt manager for handling templates."""
        self.prompt_manager = PromptManager()
        logger.debug("Prompt manager configured with centralized templates")
    
    def analyze_logs(self, logs: List[Dict[str, Any]], analysis_type: str = "anomaly_detection") -> Dict[str, Any]:
        """
        Perform concise LLM analysis on logs.
        
        Args:
            logs: List of log dictionaries
            analysis_type: Type of analysis ('anomaly_detection', 'root_cause', 'log_summary')
            
        Returns:
            Dict containing concise analysis results
        """
        try:
            if not logs:
                return {
                    "status": "error",
                    "message": "No logs provided for analysis",
                    "analysis_type": analysis_type
                }
            
            # Get prompt template from prompt manager
            prompt = self.prompt_manager.get_prompt(analysis_type)
            if prompt is None:
                return {
                    "status": "error",
                    "message": f"Unknown analysis type: {analysis_type}",
                    "available_types": self.prompt_manager.list_basic_prompts()
                }
            
            # Limit logs for concise analysis (max 20 entries)
            limited_logs = logs[:20] if len(logs) > 20 else logs
            
            # Format logs for analysis
            formatted_logs = self._format_logs_for_analysis(limited_logs)
            
            # Generate analysis
            logger.info(f"Starting concise {analysis_type} analysis on {len(limited_logs)} log entries...")
            
            # Create prompt with logs
            formatted_prompt = prompt.format(logs=formatted_logs)
            
            print(f"formatted_prompt: {formatted_prompt}")

            # Get LLM response
            response = self._get_llm_response(formatted_prompt)
            
            if response:
                logger.info(f"✅ {analysis_type} analysis completed")
                return {
                    "status": "success",
                    "analysis_type": analysis_type,
                    "log_count": len(limited_logs),
                    "total_logs": len(logs),
                    "analysis": response.strip(),
                    "model_used": self.model_name
                }
            else:
                logger.error(f"❌ {analysis_type} analysis failed - no response from LLM")
                return {
                    "status": "error",
                    "message": "Failed to get response from LLM",
                    "analysis_type": analysis_type
                }
                
        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "analysis_type": analysis_type
            }
    
    def _format_logs_for_analysis(self, logs: List[Dict[str, Any]]) -> str:
        """Format logs for concise LLM analysis."""
        try:
            formatted_logs = []
            
            for i, log in enumerate(logs, 1):
                timestamp = log.get("timestamp", "N/A")
                level = log.get("level", "INFO")
                message = log.get("message", log.get("content", ""))
                
                # Include timestamp and clearer formatting for better LLM analysis
                formatted_log = f"{i}. {timestamp} [{level}] {message}"
                formatted_logs.append(formatted_log)
                
                # Debug: check if we have parsed logs correctly
                if i <= 3:  # Log first few entries for debugging
                    logger.debug(f"Formatted log {i}: {formatted_log}")
            
            result = "\n".join(formatted_logs)
            logger.debug(f"Total formatted logs for LLM analysis: {len(formatted_logs)}")
            return result
            
        except Exception as e:
            logger.error(f"Error formatting logs: {str(e)}")
            return "Error formatting logs for analysis"
    
    def _get_llm_response(self, prompt: str) -> Optional[str]:
        """Get response from LLM with error handling."""
        try:
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, str):
                return response
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return str(response)
                
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return None
    
    # Convenience methods for specific analysis types
    def detect_anomalies(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies in logs with concise output."""
        return self.analyze_logs(logs, "anomaly_detection")
    
    def analyze_root_cause(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform root cause analysis with concise output."""
        return self.analyze_logs(logs, "root_cause")
    
    def summarize_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a concise summary of logs."""
        return self.analyze_logs(logs, "log_summary")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM model and available analysis types."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "safety_settings": "disabled",
            "available_analysis_types": self.prompt_manager.list_basic_prompts(),
            "prompt_info": self.prompt_manager.get_prompt_info()
        }
    
    def get_available_analysis_types(self) -> List[str]:
        """Get list of available analysis types."""
        return self.prompt_manager.list_basic_prompts()
    
    def add_custom_analysis_type(self, name: str, template: str, input_variables: List[str] = None) -> bool:
        """
        Add a custom analysis type with its prompt template.
        
        Args:
            name: Name for the analysis type
            template: The prompt template string
            input_variables: List of input variables (defaults to ['logs'])
            
        Returns:
            True if added successfully, False otherwise
        """
        return self.prompt_manager.add_custom_prompt(name, template, input_variables)
    
    def test_connection(self) -> bool:
        """Test LLM connection with a simple query."""
        try:
            test_prompt = "Respond with 'OK' if you can see this."
            response = self._get_llm_response(test_prompt)
            
            if response and ("ok" in response.lower() or "yes" in response.lower()):
                logger.info("✅ LLM connection test successful")
                return True
            else:
                logger.warning("⚠️ LLM connection test returned unexpected response")
                return False
                
        except Exception as e:
            logger.error(f"❌ LLM connection test failed: {str(e)}")
            return False