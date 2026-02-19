"""
MedGemma Testing Script
This script demonstrates how to use MedGemma models from Hugging Face for medical text analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

import argparse

def setup_environment(cmd_token=None):
    """Set up the Hugging Face token for authentication"""
    hf_token = cmd_token or os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("\n⚠️  No Hugging Face token found.")
        print("You can set it in your environment or pass it as an argument:")
        print("  PowerShell: $env:HF_TOKEN=\"your_token\"")
        print("  Command: python test_medgemma.py --token your_token")
        
        user_input = input("\nEnter your Hugging Face token (or press Enter to exit): ").strip()
        if user_input:
            hf_token = user_input
        else:
            return None
    return hf_token

def test_medgemma_text(model_name="google/medgemma-4b-it", hf_token=None):
    """
    Test MedGemma with text-based medical queries
    
    Args:
        model_name: The Hugging Face model ID (default: google/medgemma-4b-it)
        hf_token: Your Hugging Face access token
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}\n")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        
        print("Loading model... (this may take a while as it downloads ~8.5GB)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token
        )
        
        # Sample medical queries
        test_queries = [
            "What are the common symptoms of type 2 diabetes?",
            "Explain the mechanism of action of metformin.",
            "What are the differential diagnoses for chest pain?",
        ]
        
        print("\n" + "="*60)
        print("Running Test Queries")
        print("="*60 + "\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Query {i}: {query}")
            print("-" * 60)
            
            # Format the prompt for instruction-tuned models
            prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            
            # Decode and print response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the model's response
            response = response.split("<start_of_turn>model\n")[-1]
            
            print(f"💡 Response:\n{response}\n")
        
        print("\n✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        if "401" in str(e) or "403" in str(e):
            print("\n🔑 Authentication error. Please ensure:")
            print("   1. You have accepted the model terms at: https://huggingface.co/google/medgemma-4b-it")
            print("   2. Your HF_TOKEN is valid and has read permissions")
        return False
    
    return True

def test_model_info(model_name="google/medgemma-4b-it", hf_token=None):
    """Display model information"""
    print(f"\n{'='*60}")
    print(f"Model Information: {model_name}")
    print(f"{'='*60}\n")
    
    from huggingface_hub import model_info
    
    try:
        info = model_info(model_name, token=hf_token)
        print(f"Model ID: {info.id}")
        print(f"Author: {info.author}")
        print(f"Downloads: {info.downloads:,}")
        print(f"Likes: {info.likes:,}")
        print(f"Tags: {', '.join(info.tags[:10])}")
        
        if hasattr(info, 'card_data') and info.card_data:
            if hasattr(info.card_data, 'language'):
                print(f"Language: {info.card_data.language}")
    except Exception as e:
        print(f"Could not fetch model info: {e}")

def main():
    """Main function to run MedGemma tests"""
    parser = argparse.ArgumentParser(description="Test MedGemma models from Hugging Face")
    parser.add_argument("--token", type=str, help="Hugging Face access token")
    parser.add_argument("--model", type=str, default="google/medgemma-4b-it", help="Model ID to test")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("      MedGemma Testing Suite")
    print("="*60)
    
    # Setup
    hf_token = setup_environment(args.token)
    if not hf_token:
        print("\n⚠️  Cannot proceed without HF_TOKEN")
        return
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\n✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  CUDA not available. Using CPU (this will be slower)")
    
    model_name = args.model
    
    # Show model info
    test_model_info(model_name, hf_token)
    
    # Run tests
    test_medgemma_text(model_name, hf_token)


if __name__ == "__main__":
    main()
