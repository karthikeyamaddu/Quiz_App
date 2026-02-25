"""
Ollama Llama2-Uncensored Model Testing Script
This script loads the Llama2-Uncensored model from Ollama and tests it with dummy questions.
"""

import sys
from langchain_ollama import OllamaLLM

print("=" * 70)
print("Ollama Llama2-Uncensored Model Testing")
print("=" * 70)

# Check if Ollama is running
print("\nChecking if Ollama service is running...")
try:
    # Initialize the model
    print("Initializing Llama2-Uncensored model...")
    llama_model = OllamaLLM(model="llama2-uncensored")
    print("✓ Model initialized successfully")
except ConnectionError as e:
    print(f"ERROR: Could not connect to Ollama service!")
    print(f"Make sure Ollama is running on localhost:11434")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR initializing model: {e}")
    sys.exit(1)

# Test with dummy questions
print("\n" + "=" * 70)
print("Testing Model with Dummy Questions")
print("=" * 70)

test_questions = [
    "What is the capital of France?",
    "Explain machine learning in simple terms.",
    "What are the benefits of Python programming?",
    "How does photosynthesis work?",
    "Tell me a fact about space exploration.",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n--- Test {i} ---")
    print(f"Question: {question}")
    
    try:
        # Generate response
        print("Generating response...")
        response = llama_model.invoke(question)
        
        # Print response (limit to first 300 characters for readability)
        if len(response) > 300:
            print(f"Response: {response[:300]}...")
        else:
            print(f"Response: {response}")
        print("✓ Generation successful")
        
    except ConnectionError:
        print(f"ERROR: Connection lost to Ollama service")
        print(f"Make sure Ollama is still running")
        break
    except Exception as e:
        print(f"ERROR during generation: {e}")

print("\n" + "=" * 70)
print("✓ Model testing completed!")
print("=" * 70)
print("\nNOTE: If you see 'ERROR: Connection refused', ensure Ollama is running:")
print("  1. Download Ollama from https://ollama.ai")
print("  2. Run: ollama run llama2-uncensored")
print("  3. This will start the Ollama service on localhost:11434")
print("=" * 70)
