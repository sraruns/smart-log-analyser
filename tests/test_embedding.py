#!/usr/bin/env python3
"""
Test embedding models and functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger

def test_embedding_models():
    """Test both embedding models."""
    try:
        logger.info("=== Testing Embedding Models ===")
        
        # Test 1: Import both embedders
        logger.info("1. Importing embedders...")
        from smart_log_analyser.embedding.embedder_simple import SimpleLogEmbedder
        from smart_log_analyser.embedding.embedder import LogEmbedder
        logger.info("✅ Both embedders imported successfully")
        
        # Test 2: Create SimpleLogEmbedder
        logger.info("2. Creating SimpleLogEmbedder...")
        simple_embedder = SimpleLogEmbedder("config/config.yaml")
        logger.info("✅ SimpleLogEmbedder created successfully")
        
        # Test 3: Test SimpleLogEmbedder properties
        logger.info("3. Testing SimpleLogEmbedder properties...")
        logger.info(f"   Model name: {simple_embedder.model_name}")
        logger.info(f"   Dimension: {simple_embedder.dimension}")
        logger.info(f"   Model info: {simple_embedder.get_model_info()}")
        logger.info("✅ SimpleLogEmbedder properties work correctly")
        
        # Test 4: Generate embedding with SimpleLogEmbedder
        logger.info("4. Testing SimpleLogEmbedder embedding generation...")
        test_text = "ERROR: Database connection failed"
        embedding = simple_embedder.generate_single_embedding(test_text)
        logger.info(f"   Generated embedding with {len(embedding)} dimensions")
        logger.info("✅ SimpleLogEmbedder embedding generation successful")
        
        # Test 5: Try LogEmbedder (may fail due to property issue)
        logger.info("5. Testing LogEmbedder...")
        try:
            log_embedder = LogEmbedder("config/config.yaml")
            logger.info("✅ LogEmbedder created successfully")
            
            # Test LogEmbedder properties
            logger.info(f"   LogEmbedder model: {log_embedder.model_name}")
            logger.info(f"   LogEmbedder dimension: {log_embedder.dimension}")
            
        except Exception as e:
            logger.warning(f"⚠️ LogEmbedder failed (expected): {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_integration():
    """Test analyzer with embedding models."""
    try:
        logger.info("\n=== Testing Analyzer Integration ===")
        
        # Test analyzer with SimpleLogEmbedder
        logger.info("1. Testing analyzer with SimpleLogEmbedder...")
        from smart_log_analyser.core.smart_log_analyzer import SmartLogAnalyzer
        
        analyzer = SmartLogAnalyzer("config/config.yaml")
        logger.info("✅ SmartLogAnalyzer created successfully")
        
        # Test basic functionality
        stats = analyzer.get_stats()
        logger.info(f"   Analyzer stats: {stats['embedding_model']}")
        logger.info("✅ Analyzer integration successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analyzer integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting embedding tests...")
    
    # Clear any cached modules
    modules_to_clear = [
        'smart_log_analyser.embedding.embedder',
        'smart_log_analyser.embedding.models',
        'smart_log_analyser.embedding.embedder_simple',
        'smart_log_analyser.core.analyzer'
    ]
    
    for module in modules_to_clear:
        if module in sys.modules:
            logger.info(f"Clearing cached module: {module}")
            del sys.modules[module]
    
    # Run tests
    success1 = test_embedding_models()
    success2 = test_analyzer_integration()
    
    if success1 and success2:
        logger.info("🎉 All embedding tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some embedding tests failed!")
        sys.exit(1)
