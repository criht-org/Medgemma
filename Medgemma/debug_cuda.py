import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Force blocking to see the real error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_name = "google/medgemma-1.5-4b-it"
token = os.getenv("HF_TOKEN")

if not token:
    print("HF_TOKEN missing")
    exit()

print("Loading model in 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # Change to float16 to match strategy
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token
    )
    
    print("Testing generation...")
    input_text = "What is a tumor?"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.7,
        do_sample=True
    )
    print("Success!")
except Exception as e:
    print(f"Error caught: {e}")
