"""
T5 Fine-tuned Model Testing Script
This script loads the fine-tuned T5 model and tests it with dummy questions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Model path
model_path = "./t5-quiz-finetune"

print("=" * 60)
print("T5 Fine-tuned Model Testing")
print("=" * 60)

# Check if model files exist
if not os.path.exists(model_path):
    print(f"ERROR: Model path '{model_path}' does not exist!")
    exit(1)

print(f"\n✓ Model path found: {model_path}")

# Load tokenizer
print("\nLoading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"ERROR loading tokenizer: {e}")
    exit(1)

# Load model
print("Loading model...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit(1)

# Test with dummy questions
print("\n" + "=" * 60)
print("Testing Model with Dummy Questions")
print("=" * 60)

test_inputs = [
    "context: The capital of France is Paris. question: What is the capital of France?",
    "context: Machine Learning is a subset of Artificial Intelligence. question: What is Machine Learning?",
    "context: Python is a programming language. question: What is Python?",
]

model.eval()

with torch.no_grad():
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {test_input}")
        
        try:
            # Encode input
            encoded = tokenizer(
                test_input,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Generate output
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode output
            predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Output: {predicted_text}")
            print("✓ Generation successful")
            
        except Exception as e:
            print(f"ERROR during generation: {e}")

print("\n" + "=" * 60)
print("✓ Model testing completed successfully!")
print("=" * 60)
