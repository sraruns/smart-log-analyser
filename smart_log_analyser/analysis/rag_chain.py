from typing import List, Dict, Any, Optional, Union
import os
from loguru import logger

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores import Qdrant
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import Field

from ..vector_store.base import VectorStoreBase
from .prompt_manager import PromptManager


class LogRetriever(BaseRetriever):
    """Custom retriever for log analysis context."""
    
    vector_store: Union[VectorStore, Qdrant, VectorStoreBase] = Field(..., description="Vector store for similarity search")
    k: int = Field(default=5, description="Number of similar documents to retrieve")
    
    class Config:
        arbitrary_types_allowed = True
        
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve relevant documents for the query."""
        try:
            # Handle custom VectorStoreBase implementations
            if isinstance(self.vector_store, VectorStoreBase):
                # Use custom search method
                results = self.vector_store.search(query, n_results=self.k)
                # Convert to Document objects
                documents = []
                for result in results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {})
                    )
                    documents.append(doc)
                return documents
            else:
                # Use LangChain's similarity_search method
                results = self.vector_store.similarity_search(query, k=self.k)
                return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []


class LogAnalysisRAGChain:
    """
    RAG Chain for enhanced log analysis using LangChain.
    
    This class combines retrieval of similar historical logs with LLM generation
    to provide more accurate and context-aware log analysis.
    """
    
    def __init__(self, vector_store: Union[VectorStore, VectorStoreBase], llm_config: Dict[str, Any]):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: Vector store containing historical logs
            llm_config: Configuration for the LLM
        """
        self.vector_store = vector_store
        self.llm_config = llm_config  # Store config for later use
        self.llm = self._initialize_llm(llm_config)
        
        # Initialize the retriever with keyword arguments
        self.retriever = LogRetriever(vector_store=vector_store, k=3)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        # Initialize RAG chains
        self._initialize_chains()
        
        logger.info("LogAnalysisRAGChain initialized successfully")
    
    def _initialize_llm(self, llm_config: Dict[str, Any]):
        """Initialize the LLM model."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            llm = ChatGoogleGenerativeAI(
                model=llm_config.get('model_name', 'gemini-pro'),
                google_api_key=api_key,
                temperature=llm_config.get('temperature', 0.1),
                max_output_tokens=llm_config.get('max_output_tokens', 512),
                top_p=llm_config.get('top_p', 0.8),
                top_k=llm_config.get('top_k', 40)
            )
            
            logger.debug("LLM initialized for RAG chain")
            return llm
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    

    
    def _initialize_chains(self):
        """Initialize the RAG chains for different analysis types."""
        
        # Get prompts from centralized prompt manager
        anomaly_prompt = self.prompt_manager.get_prompt("rag_anomaly_detection")
        root_cause_prompt = self.prompt_manager.get_prompt("rag_root_cause")
        summary_prompt = self.prompt_manager.get_prompt("rag_log_summary")
        
        # Anomaly Detection Chain
        self.anomaly_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": anomaly_prompt},
            return_source_documents=True
        )
        
        # Root Cause Analysis Chain
        self.root_cause_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": root_cause_prompt},
            return_source_documents=True
        )
        
        # Log Summary Chain
        self.summary_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": summary_prompt},
            return_source_documents=True
        )
        
        logger.debug("RAG chains initialized successfully with centralized prompts")
    
    def analyze_anomalies(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform anomaly detection with RAG."""
        return self._run_analysis(logs, self.anomaly_chain, "anomaly_detection")
    
    def analyze_root_cause(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform root cause analysis with RAG."""
        return self._run_analysis(logs, self.root_cause_chain, "root_cause")
    
    def summarize_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate log summary with RAG."""
        return self._run_analysis(logs, self.summary_chain, "log_summary")
    
    def _run_analysis(self, logs: List[Dict[str, Any]], chain: RetrievalQA, analysis_type: str) -> Dict[str, Any]:
        """Run analysis using the specified RAG chain."""
        try:
            if not logs:
                return {
                    "status": "error",
                    "message": "No logs provided for analysis",
                    "analysis_type": analysis_type
                }
            
            # Format logs for analysis
            formatted_logs = self._format_logs(logs)
            
            # Run the RAG chain
            logger.info(f"Running {analysis_type} with RAG on {len(logs)} log entries...")
            
            result = chain({"query": formatted_logs})
            
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "log_count": len(logs),
                "analysis": result["result"],
                "source_documents": [doc.page_content for doc in result.get("source_documents", [])],
                "model_used": self.llm_config.get('model_name', 'gemini-pro')
            }
            
        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "analysis_type": analysis_type
            }
    
    def _format_logs(self, logs: List[Dict[str, Any]]) -> str:
        """Format logs for analysis."""
        try:
            formatted_logs = []
            
            for i, log in enumerate(logs, 1):
                timestamp = log.get("timestamp", "N/A")
                level = log.get("level", "INFO")
                message = log.get("message", log.get("content", ""))
                
                # Include more detail for better analysis
                formatted_log = f"{i}. [{timestamp}] [{level}] {message}"
                formatted_logs.append(formatted_log)
            
            return "\n".join(formatted_logs)
            
        except Exception as e:
            logger.error(f"Error formatting logs: {str(e)}")
            return "Error formatting logs for analysis"
