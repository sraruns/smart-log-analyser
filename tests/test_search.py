"""
Test script with vector store clearing and minimal LLM analysis
"""
from smart_log_analyser.main import SmartLogAnalyzer
import shutil
import os
import time

def test_with_clear_store():
    print("=== Testing with Fresh Vector Store ===")
    
    # Clear the vector store directory
    vector_store_path = "./data/vector_store"
    if os.path.exists(vector_store_path):
        print(f"Removing existing vector store at {vector_store_path}")
        shutil.rmtree(vector_store_path)
    
    # Initialize analyzer
    print("Initializing fresh analyzer...")
    analyzer = SmartLogAnalyzer()
    
    # Test log lines
    log_lines = [
        "2024-03-20 10:15:30 [INFO] User login successful",
        "2024-03-20 10:15:35 [ERROR] Database connection failed",
        "2024-03-20 10:15:40 [WARNING] High memory usage detected",
        "2024-03-20 10:15:45 [ERROR] Failed to process request",
        "2024-03-20 10:15:50 [INFO] System backup completed"
    ]
    
    # Process logs WITHOUT analysis to avoid LLM delay
    print("Processing logs (without LLM analysis)...")
    start_time = time.time()
    result = analyzer.process_logs(log_lines, analyze=False)
    process_time = time.time() - start_time
    
    print(f"Processing completed in {process_time:.2f} seconds")
    print(f"Status: {result['status']}")
    print(f"Processed {result['parsed_logs']} logs into {result['chunks']} chunks")
    
    # Test search WITHOUT analysis to avoid LLM delay
    print("\nTesting search for 'database error' (without LLM analysis)...")
    start_time = time.time()
    search_result = analyzer.search_logs("database error", analyze=False)
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.2f} seconds")
    print(f"Search status: {search_result['status']}")
    print(f"Found {len(search_result.get('results', []))} results")
    
    if search_result.get('results'):
        for i, result in enumerate(search_result['results']):
            print(f"Result {i+1}:")
            print(f"  Content: {result['content']}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Relevance: {result['relevance_score']:.4f}")
            print(f"  Chunk Type: {result['metadata'].get('chunk_type', 'unknown')}")
    else:
        print("No results found!")
    
    # Test with LLM analysis if user wants (optional)
    print(f"\nTotal time without LLM analysis: {process_time + search_time:.2f} seconds")
    
    # Optional: Test one LLM call to show the difference
    user_input = input("\nDo you want to test with LLM analysis? (y/n): ").lower().strip()
    if user_input == 'y':
        print("\nTesting search WITH LLM analysis...")
        start_time = time.time()
        search_with_analysis = analyzer.search_logs("database error", analyze=True)
        analysis_time = time.time() - start_time
        print(f"Search with analysis completed in {analysis_time:.2f} seconds")
        print("Analysis result available but not displayed for brevity.")

if __name__ == "__main__":
    test_with_clear_store()