# MedGemma Testing Guide

This folder contains scripts to test Google's MedGemma medical language models from Hugging Face.

## What is MedGemma?

MedGemma is a family of medical language models developed by Google, built on the Gemma architecture and specialized for medical and clinical applications. Available versions include:

- **MedGemma 4B**: Faster, lighter model good for most medical queries
- **MedGemma 27B**: Larger model with advanced clinical comprehension
- **Multimodal variants**: Can process both text and medical images

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster inference

### 2. Get Hugging Face Access Token

1. Create a free account at [Hugging Face](https://huggingface.co/join)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Click "New token" and create a token with **read** permissions
4. Copy the token (you'll need it in step 4)

### 3. Accept Model Terms

You must accept the usage terms for the models:
- [MedGemma 1.5 4B Terms (Multimodal)](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma 4B Terms](https://huggingface.co/google/medgemma-4b-it)
- [MedGemma 27B Terms](https://huggingface.co/google/medgemma-27b-it)

Click the "Access repository" button and accept the conditions.

### 4. Set Your Token

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_token_here"
```

**Windows (Command Prompt):**
```cmd
set HF_TOKEN=your_token_here
```

**Linux/Mac:**
```bash
export HF_TOKEN="your_token_here"
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Tests

### Basic Test
```bash
python test_medgemma.py
```

This will:
1. Load the MedGemma 4B model (default)
2. Run several sample medical queries
3. Display the model's responses

### Expected Output

The script will test queries like:
- "What are the common symptoms of type 2 diabetes?"
- "Explain the mechanism of action of metformin."
- "What are the differential diagnoses for chest pain?"

## Model Variants

### MedGemma 4B Instruction-Tuned (`google/medgemma-4b-it`)
- **Size**: ~8.5GB
- **Best for**: General medical queries, faster inference
- **RAM requirement**: ~16GB system RAM (or 8GB+ VRAM with GPU)

### MedGemma 27B Instruction-Tuned (`google/medgemma-27b-it`)
- **Size**: ~55GB
- **Best for**: Complex clinical reasoning, advanced medical analysis
- **RAM requirement**: ~64GB system RAM (or 24GB+ VRAM with GPU)

## Troubleshooting

### Authentication Errors (401/403)
- Ensure you've accepted the model terms on Hugging Face
- Verify your `HF_TOKEN` is set correctly
- Check your token has read permissions

### Out of Memory Errors
- Try using the 4B model instead of 27B
- Close other applications
- Consider using CPU inference (slower but uses less memory)

### Slow Performance
- Install CUDA if you have an NVIDIA GPU
- Use the 4B model for faster responses
- Reduce `max_new_tokens` in the generation parameters

## Customization

You can modify `test_medgemma.py` to:
- Add your own medical queries
- Adjust generation parameters (temperature, max_tokens, etc.)
- Switch between different model variants
- Save outputs to files

## Important Notes

⚠️ **Disclaimer**: MedGemma is for research and educational purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Resources

- [MedGemma Documentation](https://ai.google.dev/gemma/docs/medgemma)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [MedGemma Research Paper](https://research.google/pubs/)

## License

MedGemma models are subject to Google's Gemma Terms of Use. Please review the terms at the Hugging Face model pages.
