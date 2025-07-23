#!/usr/bin/env python3
"""
Script to list available Google Generative AI models
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

def list_available_models():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    print("Available Google Generative AI models:")
    print("=" * 50)
    
    try:
        # List all available models
        models = genai.list_models()
        
        for model in models:
            print(f"Model: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description}")
            print(f"  Supported Methods: {', '.join(model.supported_generation_methods)}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_available_models()
