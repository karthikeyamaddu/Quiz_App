"""
Script to list all available Gemini models from Google API
"""

import os
import requests
from dotenv import load_dotenv

print("=" * 70)
print("Fetching All Available Gemini Models")
print("=" * 70)

# Load environment variables
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in environment variables!")
    print("Please set GEMINI_API_KEY in your .env file")
    exit(1)

print("\n✓ API key loaded successfully")

# Fetch available models
print("\nFetching available models from Google API...")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"

try:
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        models = data.get("models", [])
        
        print(f"\n✓ Found {len(models)} models\n")
        
        # Extract model names and save to file
        model_list = []
        
        for model in models:
            model_name = model.get("name", "").replace("models/", "")
            model_display_name = model.get("displayName", "")
            model_desc = model.get("description", "")
            
            if model_name:
                model_list.append(model_name)
                print(f"• {model_name}")
                if model_display_name:
                    print(f"  Display: {model_display_name}")
                if model_desc:
                    print(f"  Description: {model_desc[:100]}...")
                print()
        
        # Save to file
        with open("gemini_models.txt", "w") as f:
            f.write("# Available Gemini Models\n")
            f.write("# Generated from Google Gemini API\n\n")
            for model in model_list:
                f.write(f"{model}\n")
        
        print("=" * 70)
        print(f"✓ Saved {len(model_list)} models to gemini_models.txt")
        print("=" * 70)
    
    else:
        print(f"ERROR: API returned status code {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"ERROR: Network error - {e}")
except Exception as e:
    print(f"ERROR: {e}")
