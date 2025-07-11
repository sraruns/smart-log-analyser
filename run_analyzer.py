from src.main import SmartLogAnalyzer
from loguru import logger

def main():
    # Initialize the analyzer
    analyzer = SmartLogAnalyzer()
    
    # Example log lines
    log_lines = [
        "2024-03-20 10:15:30 [INFO] User login successful",
        "2024-03-20 10:15:35 [ERROR] Database connection failed",
        "2024-03-20 10:15:40 [WARNING] High memory usage detected",
        "2024-03-20 10:15:45 [ERROR] Failed to process request",
        "2024-03-20 10:15:50 [INFO] System backup completed"
    ]
    
    # Process logs with analysis
    logger.info("Processing logs...")
    result = analyzer.process_logs(log_lines, analyze=True)
    
    # Print processing results
    print("\nProcessing Results:")
    print(f"Status: {result['status']}")
    print(f"Parsed Logs: {result['parsed_logs']}")
    print(f"Chunks: {result['chunks']}")
    
    # Print analysis results
    if result.get('analysis'):
        print("\nAnalysis Results:")
        for analysis_type, analysis_result in result['analysis'].items():
            print(f"\n{analysis_type.upper()} Analysis:")
            print(analysis_result['result'])
    
    # Search for similar logs
    logger.info("\nSearching for similar logs...")
    search_result = analyzer.search_logs("database error", analyze=True)
    
    # Print search results
    print("\nSearch Results:")
    for result in search_result['results']:
        print(f"Found: {result['content']} (Distance: {result['distance']})")
    
    if search_result.get('analysis'):
        print("\nSearch Analysis:")
        print(search_result['analysis']['result'])

if __name__ == "__main__":
    main() 