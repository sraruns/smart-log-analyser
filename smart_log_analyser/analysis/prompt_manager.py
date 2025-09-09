from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from loguru import logger


class PromptManager:
    """
    Centralized prompt management for all log analysis tasks.
    
    Features:
    - Unified prompt storage for basic LLM and RAG analysis
    - Easy customization and extension
    - Consistent formatting across analysis types
    - Template validation and error handling
    """
    
    def __init__(self):
        """Initialize the prompt manager with all analysis templates."""
        self.prompts = {}
        self._setup_basic_prompts()
        self._setup_rag_prompts()
        logger.debug("Prompt manager initialized with all templates")
    
    def _setup_basic_prompts(self):
        """Setup basic prompt templates for simple LLM analysis."""
        
        # Basic Anomaly Detection Prompt
        self.prompts["anomaly_detection"] = PromptTemplate(
            input_variables=["logs"],
            template="""Analyze these logs for anomalies and issues. Look specifically for:
- ERROR level messages
- WARNING level messages  
- CRITICAL level messages
- Failed operations
- Timeouts and connection issues
- Resource problems (memory, disk space)

LOGS TO ANALYZE:
{logs}

Format response as:
ANOMALIES: [List each ERROR/WARNING/CRITICAL found with severity: HIGH/MEDIUM/LOW]
PATTERNS: [Notable patterns like repeated errors, escalating issues]
ACTIONS: [Immediate actions needed based on the issues found]

Be thorough - do not miss any ERROR, WARNING, or CRITICAL entries."""
        )
        
        # Basic Root Cause Analysis Prompt
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
        
        # Basic Log Summary Prompt
        self.prompts["log_summary"] = PromptTemplate(
            input_variables=["logs"],
            template="""Summarize these logs and identify all issues:

{logs}

Carefully examine EVERY log entry and identify:
- All ERROR entries and their details
- All WARNING entries and their details  
- All CRITICAL entries and their details
- System status based on the severity of issues found

Format response as:
STATUS: [HEALTHY if no errors/warnings, WARNING if warnings present, CRITICAL if errors/critical present]
KEY EVENTS: [List the most important events, especially errors and warnings]
ISSUES: [List EVERY error and warning found - be comprehensive]
ACTIONS: [Recommended actions based on the issues identified]

Do not miss any ERROR, WARNING, or CRITICAL log entries."""
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
    
    def _setup_rag_prompts(self):
        """Setup enhanced RAG prompt templates with context integration."""
        
        # RAG Anomaly Detection Prompt
        self.prompts["rag_anomaly_detection"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert log analyst specializing in anomaly detection. 
Use the following context from similar historical logs to help analyze the current logs.

HISTORICAL CONTEXT:
{context}

CURRENT LOGS TO ANALYZE:
{question}

INSTRUCTIONS:
1. Carefully examine each log entry for errors, warnings, and anomalies
2. Pay special attention to:
   - ERROR and CRITICAL level messages
   - Database connection issues
   - Payment processing failures
   - Memory/disk space warnings
   - Service timeouts and unavailability
3. Use the historical context to identify patterns and classify issues accurately
4. For each issue found, provide severity level and evidence

RESPONSE FORMAT:
## ANOMALY ANALYSIS

**OVERALL STATUS:** [HEALTHY/WARNING/CRITICAL]

**DETECTED ISSUES:**
1. **[ISSUE_TYPE]** - Severity: [HIGH/MEDIUM/LOW]
   - Description: [Brief description]
   - Evidence: [Specific log entries]
   - Similar Historical Pattern: [Reference to context if applicable]

**SUMMARY:**
- Total Issues Found: [NUMBER]
- Critical Issues: [NUMBER]
- Immediate Actions Required: [LIST]

Be thorough and accurate. Do not miss obvious errors or warnings."""
        )
        
        # RAG Root Cause Analysis Prompt
        self.prompts["rag_root_cause"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert system administrator performing root cause analysis.
Use the following context from similar historical incidents to help identify root causes.

HISTORICAL CONTEXT:
{context}

CURRENT LOGS TO ANALYZE:
{question}

INSTRUCTIONS:
1. Identify the primary issues in the logs
2. Trace the sequence of events leading to problems
3. Use historical context to identify common root causes
4. Provide actionable recommendations
5. **KEEP YOUR RESPONSE CONCISE - MAXIMUM 500 WORDS**

RESPONSE FORMAT:
## ROOT CAUSE ANALYSIS

**PRIMARY ISSUES:** [List main problems - be brief]

**ROOT CAUSE:** [Primary cause - 1-2 sentences]

**EVIDENCE:** [Key log entries - 2-3 points max]

**RECOMMENDATIONS:** [3-4 key actions - be specific and brief]

Keep it concise and actionable."""
        )
        
        # RAG Log Summary Prompt
        self.prompts["rag_log_summary"] = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert log analyst. Provide a comprehensive summary of the logs.
Use the historical context to better understand patterns and significance.

HISTORICAL CONTEXT:
{context}

CURRENT LOGS TO ANALYZE:
{question}

INSTRUCTIONS:
1. Summarize key events and their significance
2. Highlight all errors and warnings (don't miss any)
3. Assess overall system health
4. Use historical context for better interpretation

RESPONSE FORMAT:
## LOG SUMMARY

**SYSTEM STATUS:** [HEALTHY/WARNING/CRITICAL]

**KEY EVENTS:**
- [List significant events with timestamps]

**ERRORS AND WARNINGS:**
- **Errors:** [Count and types - be specific]
- **Warnings:** [Count and types - be specific]
- **Critical Issues:** [Any critical problems]

**PATTERNS OBSERVED:**
- [Notable patterns or trends]

**RECOMMENDATIONS:**
- [Key actions needed]

Ensure you identify ALL errors and warnings in the logs."""
        )
    
    def get_prompt(self, analysis_type: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template for the specified analysis type.
        
        Args:
            analysis_type: Type of analysis (e.g., 'anomaly_detection', 'rag_anomaly_detection')
            
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
    
    def list_basic_prompts(self) -> List[str]:
        """Get list of basic (non-RAG) prompt templates."""
        return [name for name in self.prompts.keys() if not name.startswith('rag_')]
    
    def list_rag_prompts(self) -> List[str]:
        """Get list of RAG prompt templates."""
        return [name for name in self.prompts.keys() if name.startswith('rag_')]
    
    def format_prompt(self, analysis_type: str, **kwargs) -> Optional[str]:
        """
        Format a prompt with the provided variables.
        
        Args:
            analysis_type: Type of analysis prompt
            **kwargs: Variables to format the prompt with
            
        Returns:
            Formatted prompt string or None if template not found
        """
        prompt_template = self.get_prompt(analysis_type)
        if prompt_template is None:
            return None
        
        try:
            return prompt_template.format(**kwargs)
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
            # Test format with dummy data based on input variables
            if 'context' in prompt_template.input_variables:
                # RAG prompt
                prompt_template.format(context="Test context", question="Test logs")
            else:
                # Basic prompt
                prompt_template.format(logs="Test log entry")
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
        basic_prompts = self.list_basic_prompts()
        rag_prompts = self.list_rag_prompts()
        
        return {
            "total_prompts": len(self.prompts),
            "basic_prompts": basic_prompts,
            "rag_prompts": rag_prompts,
            "available_types": list(self.prompts.keys()),
            "basic_input_variables": ["logs"],
            "rag_input_variables": ["context", "question"],
            "custom_prompts_supported": True
        }
