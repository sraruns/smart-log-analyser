from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from loguru import logger


class PromptManager:
    """
    Centralized prompt management for log analysis tasks.
    
    Features:
    - Modular prompt templates
    - Easy customization and extension
    - Consistent formatting across analysis types
    - Template validation and error handling
    """
    
    def __init__(self):
        """Initialize the prompt manager with default templates."""
        self.prompts = {}
        self._setup_default_prompts()
        logger.debug("Prompt manager initialized with default templates")
    
    def _setup_default_prompts(self):
        """Setup default prompt templates for log analysis."""
        
        # Anomaly Detection Prompt
        self.prompts["anomaly_detection"] = PromptTemplate(
            input_variables=["logs"],
            template="""Analyze these logs for anomalies. Provide CONCISE key insights only:

{logs}

Format response as:
ANOMALIES: [List 2-3 key issues with severity: HIGH/MEDIUM/LOW]
PATTERNS: [1-2 notable patterns]
ACTIONS: [1-2 immediate actions needed]"""
        )
        
        # Root Cause Analysis Prompt
        self.prompts["root_cause"] = PromptTemplate(
            input_variables=["logs"],
            template="""Identify root causes in these logs. Be CONCISE:

{logs}

Format response as:
PRIMARY ISSUE: [Main problem identified]
ROOT CAUSE: [Most likely cause]
EVIDENCE: [Key supporting log entries]
FIX: [Primary recommended action]"""
        )
        
        # Log Summary Prompt
        self.prompts["log_summary"] = PromptTemplate(
            input_variables=["logs"],
            template="""Summarize these logs concisely:

{logs}

Format response as:
STATUS: [HEALTHY/WARNING/CRITICAL]
KEY EVENTS: [2-3 most important events]
ERRORS: [Count and types]
RECOMMENDATIONS: [1-2 key actions]"""
        )
        
        # Performance Analysis Prompt
        self.prompts["performance_analysis"] = PromptTemplate(
            input_variables=["logs"],
            template="""Analyze performance indicators in these logs:

{logs}

Format response as:
PERFORMANCE STATUS: [GOOD/DEGRADED/POOR]
BOTTLENECKS: [Identified performance issues]
METRICS: [Key performance indicators found]
OPTIMIZATION: [Performance improvement suggestions]"""
        )
        
        # Security Analysis Prompt
        self.prompts["security_analysis"] = PromptTemplate(
            input_variables=["logs"],
            template="""Analyze these logs for security concerns:

{logs}

Format response as:
SECURITY STATUS: [SECURE/SUSPICIOUS/CRITICAL]
THREATS: [Potential security issues identified]
INDICATORS: [Security-related log patterns]
RESPONSE: [Recommended security actions]"""
        )
    
    def get_prompt(self, analysis_type: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template for the specified analysis type.
        
        Args:
            analysis_type: Type of analysis (e.g., 'anomaly_detection', 'root_cause')
            
        Returns:
            PromptTemplate object or None if not found
        """
        if analysis_type not in self.prompts:
            logger.warning(f"Prompt template '{analysis_type}' not found. Available: {list(self.prompts.keys())}")
            return None
        
        return self.prompts[analysis_type]
    
    def add_custom_prompt(self, name: str, template: str, input_variables: List[str] = None) -> bool:
        """
        Add a custom prompt template.
        
        Args:
            name: Name for the prompt template
            template: The prompt template string
            input_variables: List of input variables (defaults to ['logs'])
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            if input_variables is None:
                input_variables = ["logs"]
            
            self.prompts[name] = PromptTemplate(
                input_variables=input_variables,
                template=template
            )
            
            logger.info(f"Added custom prompt template: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom prompt '{name}': {str(e)}")
            return False
    
    def list_available_prompts(self) -> List[str]:
        """
        Get a list of all available prompt templates.
        
        Returns:
            List of prompt template names
        """
        return list(self.prompts.keys())
    
    def format_prompt(self, analysis_type: str, logs: str) -> Optional[str]:
        """
        Format a prompt with the provided logs.
        
        Args:
            analysis_type: Type of analysis prompt
            logs: Formatted log string
            
        Returns:
            Formatted prompt string or None if template not found
        """
        prompt_template = self.get_prompt(analysis_type)
        if prompt_template is None:
            return None
        
        try:
            return prompt_template.format(logs=logs)
        except Exception as e:
            logger.error(f"Error formatting prompt '{analysis_type}': {str(e)}")
            return None
    
    def validate_prompt(self, analysis_type: str) -> bool:
        """
        Validate that a prompt template exists and is properly formatted.
        
        Args:
            analysis_type: Type of analysis prompt to validate
            
        Returns:
            True if prompt is valid, False otherwise
        """
        prompt_template = self.get_prompt(analysis_type)
        if prompt_template is None:
            return False
        
        try:
            # Test format with dummy data
            test_logs = "Test log entry"
            prompt_template.format(logs=test_logs)
            return True
        except Exception as e:
            logger.error(f"Prompt validation failed for '{analysis_type}': {str(e)}")
            return False
    
    def get_prompt_info(self) -> Dict[str, Any]:
        """
        Get information about all available prompts.
        
        Returns:
            Dictionary with prompt information
        """
        return {
            "total_prompts": len(self.prompts),
            "available_types": list(self.prompts.keys()),
            "default_input_variables": ["logs"],
            "custom_prompts_supported": True
        }
