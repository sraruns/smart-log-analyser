#!/usr/bin/env python3
"""
Enhanced Smart Log Analyzer with RAG Chain Implementation.

This demo showcases:
- Log processing with embedding generation
- RAG-enhanced LLM analysis using LangChain
- Improved error detection and classification
- Context-aware analysis using historical patterns
"""

import sys
import os
from typing import List, Dict, Any
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from smart_log_analyser.core.smart_log_analyzer import SmartLogAnalyzer
from smart_log_analyser.analysis.rag_chain import LogAnalysisRAGChain
from smart_log_analyser.data.sample_logs import SampleLogs


def demonstrate_rag_enhanced_analysis(sample_logs):
    """Demonstrate the RAG-enhanced Smart Log Analyzer."""
    try:
        logger.info("=== RAG-Enhanced Smart Log Analyzer Demo ===")
        
        # Initialize analyzer
        analyzer = SmartLogAnalyzer("config/config.yaml")
        

        logger.info(f"📝 Processing {len(sample_logs)} sample log entries...")
        
        # First, process logs to populate the vector store (for RAG)
        result = analyzer.process_logs_for_vector_storage(sample_logs)
        
        if result["status"] == "success":
            logger.info(f"✅ Processed {result['processed_logs']} logs into {result['chunks_created']} chunks")
            logger.info(f"📊 Added {result['embeddings_added']} embeddings to vector store")
            logger.info(f"🤖 Using embedding model: {result['embedding_model']['model_name']}")
        else:
            logger.error(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
            return False
        
        # Initialize RAG Chain with the vector store
        logger.info("🔗 Initializing RAG Chain...")
        
        # Get LLM config from analyzer
        with open("config/config.yaml", 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        llm_config = config.get('llm', {})
        
        # Check for Google API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            logger.warning("⚠️  GOOGLE_API_KEY not found or not set!")
            logger.info("📝 To enable LLM analysis:")
            logger.info("   1. Get your API key from: https://makersuite.google.com/app/apikey")
            logger.info("   2. Set it in .env file: GOOGLE_API_KEY=your_actual_key")
            logger.info("   3. Or export it: export GOOGLE_API_KEY=your_actual_key")
            logger.info("🚀 Continuing with basic analysis (without RAG)...")
            return True
        
     
        # Create RAG chain
        rag_chain = LogAnalysisRAGChain(analyzer.vector_store, llm_config)
        
        logger.info("✅ RAG Chain initialized successfully")
        
        # Run RAG analysis demo
        run_rag_analysis_demo(rag_chain, analyzer, sample_logs)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_basic_llm_analysis(analyzer, sample_logs):
    """Run basic LLM analysis without RAG."""
    logger.info("\n🔍 Running Basic LLM Analysis (no RAG):")
    basic_analysis_result = analyzer.analyze_logs_with_llm(
        sample_logs, 
        analysis_types=["anomaly_detection", "log_summary"]
    )
    
    if basic_analysis_result["status"] == "success":
        logger.info("✅ Basic LLM analysis completed")
        logger.info(f"📊 Analyzed {basic_analysis_result['processed_logs']} log entries")
        logger.info(f"🔧 Analysis types: {basic_analysis_result['analysis_types']}")
        
        # Show basic analysis results
        print("\n" + "="*80)
        print("BASIC LLM ANALYSIS RESULTS (NO RAG):")
        print("="*80)
        for analysis_type, result in basic_analysis_result["analysis"].items():
            if result.get("status") == "success":
                print(f"\n--- {analysis_type.upper()} ---")
                print(result["analysis"])
        print("="*80 + "\n")
    else:
        logger.warning(f"⚠️ Basic analysis failed: {basic_analysis_result.get('message', 'Unknown error')}")

def run_rag_analysis_demo(rag_chain, analyzer, sample_logs):
    """Run demonstration of RAG-enhanced analysis capabilities."""
    # Demonstrate RAG-Enhanced Analysis
    logger.info("\n🧠 Running RAG-Enhanced Analysis (with historical context):")
    
    # Convert string logs to dict format for RAG analysis
    sample_logs_dict = []
    for log_line in sample_logs:
        # Parse the string format: "timestamp level: message"
        parts = log_line.split(" ", 2)
        if len(parts) >= 3:
            timestamp = f"{parts[0]} {parts[1]}"
            level_and_message = parts[2]
            if ":" in level_and_message:
                level, message = level_and_message.split(":", 1)
                sample_logs_dict.append({
                    "timestamp": timestamp,
                    "level": level.strip(),
                    "message": message.strip()
                })
    
    # Test 1: Anomaly Detection with RAG
    logger.info("\n  1. 🔍 Anomaly Detection Analysis:")
    try:
        anomaly_result = rag_chain.analyze_anomalies(sample_logs_dict)
        if anomaly_result["status"] == "success":
            logger.info("     ✅ Anomaly detection completed")
            logger.info(f"     📋 Analysis type: {anomaly_result['analysis_type']}")
            logger.info(f"     📊 Analyzed {anomaly_result['log_count']} log entries")
            logger.info(f"     🤖 Model used: {anomaly_result['model_used']}")
            logger.info(f"     📚 Retrieved {len(anomaly_result.get('source_documents', []))} context documents")
            
            # Display the analysis result
            print("\n" + "="*80)
            print("ANOMALY DETECTION RESULTS:")
            print("="*80)
            print(anomaly_result["analysis"])
            print("="*80 + "\n")
            
        else:
            logger.warning(f"     ⚠️ Anomaly detection failed: {anomaly_result.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"     ❌ Anomaly detection error: {str(e)}")
    
    # Test 2: Root Cause Analysis with RAG
    logger.info("  2. 🔍 Root Cause Analysis:")
    try:
        root_cause_result = rag_chain.analyze_root_cause(sample_logs_dict)
        if root_cause_result["status"] == "success":
            logger.info("     ✅ Root cause analysis completed")
            logger.info(f"     📋 Analysis type: {root_cause_result['analysis_type']}")
            logger.info(f"     📊 Analyzed {root_cause_result['log_count']} log entries")
            logger.info(f"     📚 Retrieved {len(root_cause_result.get('source_documents', []))} context documents")
            
            # Display the analysis result
            print("\n" + "="*80)
            print("ROOT CAUSE ANALYSIS RESULTS:")
            print("="*80)
            print(root_cause_result["analysis"])
            print("="*80 + "\n")
            
        else:
            logger.warning(f"     ⚠️ Root cause analysis failed: {root_cause_result.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"     ❌ Root cause analysis error: {str(e)}")
    
    # Test 3: Log Summary with RAG
    logger.info("  3. 📋 Log Summary Analysis:")
    try:
        summary_result = rag_chain.summarize_logs(sample_logs_dict)
        if summary_result["status"] == "success":
            logger.info("     ✅ Log summary completed")
            logger.info(f"     📋 Analysis type: {summary_result['analysis_type']}")
            logger.info(f"     📊 Analyzed {summary_result['log_count']} log entries")
            logger.info(f"     📚 Retrieved {len(summary_result.get('source_documents', []))} context documents")
            
            # Display the analysis result
            print("\n" + "="*80)
            print("LOG SUMMARY RESULTS:")
            print("="*80)
            print(summary_result["analysis"])
            print("="*80 + "\n")
            
        else:
            logger.warning(f"     ⚠️ Log summary failed: {summary_result.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"     ❌ Log summary error: {str(e)}")
    
    # Display final statistics
    logger.info("📈 Final Statistics:")
    stats = analyzer.get_stats()
    vector_stats = stats.get("vector_store", {})
    embedding_stats = stats.get("embedding_model", {})
    
    logger.info(f"   Total embeddings: {vector_stats.get('total_embeddings', 'Unknown')}")
    logger.info(f"   Vector store version: {vector_stats.get('version', 'Unknown')}")
    logger.info(f"   Last updated: {vector_stats.get('last_updated', 'Unknown')}")
    
    logger.info("🤖 Embedding Model Information:")
    logger.info(f"   Current model: {embedding_stats.get('model_name', 'Unknown')}")
    logger.info(f"   Model type: {embedding_stats.get('type', 'Unknown')}")
    logger.info(f"   Dimension: {embedding_stats.get('dimension', 'Unknown')}")
    
    logger.info("✅ RAG-enhanced demo completed successfully!")


def demonstrate_embedding_comparison():
    """Demonstrate embedding model comparison capabilities."""
    try:
        logger.info("=== Embedding Model Comparison ===")
        
        analyzer = SmartLogAnalyzer("config/config.yaml")
        comparison = analyzer.get_embedding_comparison()
        
        logger.info("📊 Embedding Model Comparison Results:")
        logger.info(f"   Current model: {comparison.get('current_model', {})}")
        logger.info(f"   Available models: {comparison.get('available_models', [])}")
        logger.info(f"   Recommendation: {comparison.get('recommendation', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding comparison failed: {str(e)}")
        return False


def main():
    """Main function to run the enhanced demo."""
    try:
        logger.info("🚀 Starting RAG-Enhanced Smart Log Analyzer Demo...")
        
        # Initialize analyzer
        analyzer = SmartLogAnalyzer("config/config.yaml")
      
        # Get sample logs from SampleLogs class
        sample_logs = SampleLogs.get_sample_logs()

        elasticsearch_logs = SampleLogs.get_logs_from_elasticsearch(None, "logs-*", 1000, "1h")

        logs_to_analyze = elasticsearch_logs

        # Run basic LLM analysis
        run_basic_llm_analysis(analyzer, logs_to_analy)

        # Run main demonstration
        demo_success = demonstrate_rag_enhanced_analysis(logs_to_analyze)
        
        # Run embedding comparison
        comparison_success = demonstrate_embedding_comparison()
        
        if demo_success and comparison_success:
            logger.info("🎉 All demos completed successfully!")
            logger.info("💡 The Smart Log Analyzer with RAG is ready for production use!")
            logger.info("📚 Features demonstrated:")
            logger.info("   - Log processing and chunking")
            logger.info("   - Embedding generation (auto-selected model)")
            logger.info("   - Vector storage (Qdrant/Chroma)")
            logger.info("   - RAG-enhanced LLM analysis with historical context")
            logger.info("   - Improved error detection and classification")
            logger.info("   - Context-aware anomaly detection")
            logger.info("   - Comprehensive error handling and logging")
            return 0
        else:
            logger.error("💥 Some demos failed. Check the logs above for details.")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())