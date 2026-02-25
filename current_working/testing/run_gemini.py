"""
Google Gemini API Testing Script for Quiz Generation
This script tests the Gemini API (gemini-2.0-flash) for generating quizzes.
"""

import os
import sys
import json
import requests
import time
from dotenv import load_dotenv

print("=" * 70)
print("Google Gemini API Testing for Quiz Generation")
print("=" * 70)

# Load environment variables
print("\nLoading environment variables...")
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in environment variables!")
    print("\nHow to set up:")
    print("1. Create a .env file in the current_working folder")
    print("2. Add: GEMINI_API_KEY=your_api_key_here")
    print("3. Or set the environment variable: set GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

print("✓ API key loaded successfully")

# Load available models from file
print("\nLoading best models from best_gemini_models.txt...")
AVAILABLE_MODELS = []

if os.path.exists("best_gemini_models.txt"):
    with open("best_gemini_models.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                AVAILABLE_MODELS.append(line)
    print(f"✓ Loaded {len(AVAILABLE_MODELS)} best models from best_gemini_models.txt")
else:
    print("⚠️  best_gemini_models.txt not found!")
    print("Run: python list_gemini_models.py")
    print("Using default best models instead...")
    AVAILABLE_MODELS = [
        "gemini-2.5-flash",
        "gemma-3-27b-it",
        "gemma-3-12b-it",
        "gemini-flash-latest",
    ]
    print(f"✓ Using {len(AVAILABLE_MODELS)} default best models")

print(f"\nAvailable best models to test:")
for model in AVAILABLE_MODELS:  # Show all best models
    print(f"  • {model}")

class GeminiQuizGenerator:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def generate(self, prompt: str, max_tokens: int = 500, max_retries: int = 3) -> str:
        """Generate content using Gemini API with retry logic"""
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            }
        }
        
        url = f"{self.endpoint}?key={self.api_key}"
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                
                elif response.status_code == 429:
                    # Rate limited - try to extract retry delay
                    error_data = response.json()
                    retry_delay = 60  # Default 60 seconds
                    
                    # Try to extract retry delay from response
                    if "details" in error_data.get("error", {}):
                        for detail in error_data["error"]["details"]:
                            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                                retry_delay_str = detail.get("retryDelay", "60s")
                                if "s" in retry_delay_str:
                                    try:
                                        retry_delay = int(float(retry_delay_str.replace("s", "")))
                                    except:
                                        pass
                    
                    if retry_count < max_retries - 1:
                        print(f"  ⏳ Rate limited (429). Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay + 5)  # Add 5s buffer
                        retry_count += 1
                        continue
                    else:
                        raise Exception(f"API Error 429 (Rate Limited): Free tier quota exceeded. Please wait before retrying or upgrade to a paid plan.")
                
                else:
                    raise Exception(f"API Error {response.status_code}: {response.text}")
            
            except requests.exceptions.RequestException as e:
                raise Exception(f"Network error: {e}")
        
        raise Exception("Max retries exceeded")

# Initialize generator
print(f"\nInitializing Gemini model testing...")
print("=" * 70)
print("Testing Quiz Generation with Different Gemini Models")
print("=" * 70)

# Single test prompt for all models
test_prompt = "Generate a multiple choice quiz question about Python with 4 options and mark the correct answer."

results = {
    "successful": [],
    "quota_exceeded": [],
    "errors": []
}

for i, model_name in enumerate(AVAILABLE_MODELS, 1):
    print(f"\n[{i}/{len(AVAILABLE_MODELS)}] Testing: {model_name}")
    print("-" * 70)
    
    try:
        generator = GeminiQuizGenerator(GEMINI_API_KEY, model_name)
        print(f"✓ Generator initialized")
        print(f"Prompt: {test_prompt[:60]}...")
        print("Generating response...")
        
        response = generator.generate(test_prompt, max_tokens=300, max_retries=1)
        
        # Print response
        if len(response) > 300:
            print(f"✓ Response (truncated):\n{response[:300]}...\n")
        else:
            print(f"✓ Response:\n{response}\n")
        
        results["successful"].append(model_name)
        print(f"✓ {model_name} - SUCCESS")
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Rate Limited" in error_msg or "quota" in error_msg.lower():
            results["quota_exceeded"].append(model_name)
            print(f"⏳ {model_name} - QUOTA EXCEEDED")
            print(f"   {error_msg[:100]}...")
        else:
            results["errors"].append((model_name, error_msg))
            print(f"❌ {model_name} - ERROR")
            print(f"   {error_msg[:100]}...")

print("\n" + "=" * 70)
print("Testing Summary")
print("=" * 70)
print(f"\nTotal models tested: {len(AVAILABLE_MODELS)}")
print(f"✓ Successful: {len(results['successful'])}")
print(f"⏳ Quota Exceeded: {len(results['quota_exceeded'])}")
print(f"❌ Errors: {len(results['errors'])}")

if results["successful"]:
    print(f"\n✓ WORKING MODELS:")
    for model in results["successful"]:
        print(f"  • {model}")

if results["quota_exceeded"]:
    print(f"\n⏳ QUOTA EXCEEDED (try later):")
    for model in results["quota_exceeded"]:
        print(f"  • {model}")

if results["errors"]:
    print(f"\n❌ ERRORS:")
    for model, error in results["errors"]:
        print(f"  • {model}")
        print(f"    {error[:80]}...")

print("\n" + "=" * 70)
print("Model Testing Complete!")
print("=" * 70)
print("\nRECOMMENDATIONS FOR QUIZ GENERATION:")
if results["successful"]:
    print(f"\n✓ Best model for quiz generation: {results['successful'][0]}")
    print(f"\nAll working models ({len(results['successful'])} available):")
    for model in results['successful']:
        print(f"  • {model}")
else:
    print("\n⚠️  No models successfully tested")
    
print("\nFREE TIER QUOTAS (per model):")
print("  • 15 requests per minute")
print("  • 1500 requests per day")
print("  • Limited input tokens per minute")

print("\nTO CUSTOMIZE MODEL TESTING:")
print("  1. Edit best_gemini_models.txt to add/remove models")
print("  2. Run run_gemini.py to test your custom list")
print("  3. Or run list_gemini_models.py to fetch all available models")
print("\nTO USE RECOMMENDED MODELS:")
print("  • Use the best models list for faster, more reliable results")
print("  • Or use run_t5.py and run_ollama.py (no API quotas)")
print("=" * 70)
