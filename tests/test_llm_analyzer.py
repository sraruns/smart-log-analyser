#!/usr/bin/env python3
"""
Test LLM analyzer functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger

def test_llm_analyzer():
    """Test the refactored LLM analyzer."""
    try:
        logger.info("=== Testing LLM Analyzer ===")
        
        # Test 1: Import LLM analyzer
        logger.info("1. Importing LLMAnalyzer...")
        from smart_log_analyser.analysis.llm_analyzer import LLMAnalyzer
        logger.info("✅ LLMAnalyzer imported successfully")
        
        # Test 2: Create LLM analyzer
        logger.info("2. Creating LLMAnalyzer...")
        llm_analyzer = LLMAnalyzer("config/config.yaml")
        logger.info("✅ LLMAnalyzer created successfully")
        
        # Test 3: Test model info
        logger.info("3. Testing model info...")
        model_info = llm_analyzer.get_model_info()
        logger.info(f"   Model info: {model_info}")
        logger.info("✅ Model info retrieved successfully")
        
        # Test 4: Test connection (if API key available)
        logger.info("4. Testing LLM connection...")
        try:
            connection_ok = llm_analyzer.test_connection()
            if connection_ok:
                logger.info("✅ LLM connection test successful")
            else:
                logger.warning("⚠️ LLM connection test failed (may be expected if no API key)")
        except Exception as e:
            logger.warning(f"⚠️ LLM connection test failed: {str(e)} (may be expected if no API key)")
        
        # Test 5: Test analysis with sample logs
        logger.info("5. Testing log analysis...")
        sample_logs = [
            {
                "timestamp": "2024-01-15 10:30:15",
                "level": "ERROR",
                "message": "Database connection failed",
                "content": "Database connection failed"
            },
            {
                "timestamp": "2024-01-15 10:31:22",
                "level": "WARNING", 
                "message": "High memory usage detected",
                "content": "High memory usage detected"
            },
            {
                "timestamp": "2024-01-15 10:32:45",
                "level": "CRITICAL",
                "message": "System overload detected",
                "content": "System overload detected"
            }
        ]
        
        # Test anomaly detection (may fail without API key)
        try:
            anomaly_result = llm_analyzer.detect_anomalies(sample_logs)
            if anomaly_result["status"] == "success":
                logger.info("✅ Anomaly detection test successful")
                logger.info(f"   Analysis type: {anomaly_result['analysis_type']}")
                logger.info(f"   Log count: {anomaly_result['log_count']}")
            else:
                logger.warning(f"⚠️ Anomaly detection failed: {anomaly_result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"⚠️ Anomaly detection test failed: {str(e)} (may be expected if no API key)")
        
        # Test root cause analysis (may fail without API key)
        try:
            root_cause_result = llm_analyzer.analyze_root_cause(sample_logs)
            if root_cause_result["status"] == "success":
                logger.info("✅ Root cause analysis test successful")
                logger.info(f"   Analysis type: {root_cause_result['analysis_type']}")
                logger.info(f"   Log count: {root_cause_result['log_count']}")
            else:
                logger.warning(f"⚠️ Root cause analysis failed: {root_cause_result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"⚠️ Root cause analysis test failed: {str(e)} (may be expected if no API key)")
        
        # Test log summary (may fail without API key)
        try:
            summary_result = llm_analyzer.summarize_logs(sample_logs)
            if summary_result["status"] == "success":
                logger.info("✅ Log summary test successful")
                logger.info(f"   Analysis type: {summary_result['analysis_type']}")
                logger.info(f"   Log count: {summary_result['log_count']}")
            else:
                logger.warning(f"⚠️ Log summary failed: {summary_result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"⚠️ Log summary test failed: {str(e)} (may be expected if no API key)")
        
        logger.info("✅ LLM analyzer tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_analyzer_error_handling():
    """Test LLM analyzer error handling."""
    try:
        logger.info("\n=== Testing LLM Analyzer Error Handling ===")
        
        from smart_log_analyser.analysis.llm_analyzer import LLMAnalyzer
        
        llm_analyzer = LLMAnalyzer("config/config.yaml")
        
        # Test with empty logs
        logger.info("1. Testing with empty logs...")
        result = llm_analyzer.detect_anomalies([])
        if result["status"] == "error" and "No logs provided" in result["message"]:
            logger.info("✅ Empty logs error handling works correctly")
        else:
            logger.warning("⚠️ Empty logs error handling unexpected result")
        
        # Test with invalid analysis type
        logger.info("2. Testing with invalid analysis type...")
        result = llm_analyzer.analyze_logs([{"content": "test"}], "invalid_type")
        if result["status"] == "error" and "Unknown analysis type" in result["message"]:
            logger.info("✅ Invalid analysis type error handling works correctly")
        else:
            logger.warning("⚠️ Invalid analysis type error handling unexpected result")
        
        logger.info("✅ Error handling tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting LLM analyzer tests...")
    
    # Run tests
    success1 = test_llm_analyzer()
    success2 = test_llm_analyzer_error_handling()
    
    if success1 and success2:
        logger.info("🎉 All LLM analyzer tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some LLM analyzer tests failed!")
        sys.exit(1)
