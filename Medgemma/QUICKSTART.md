# Quick Start Guide - MedGemma Testing

## ✅ Setup Complete!

All dependencies have been installed. Follow these steps to test MedGemma:

## Step 1: Get Your Hugging Face Token

1. **Create/Login** to your Hugging Face account: https://huggingface.co/join
2. **Get a token**: Go to https://huggingface.co/settings/tokens
3. **Create new token** with **read** permissions
4. **Copy** the token

## Step 2: Accept Model Terms

Visit these pages and click "Access repository":
- https://huggingface.co/google/medgemma-1.5-4b-it
- https://huggingface.co/google/medgemma-4b-it
- https://huggingface.co/google/medgemma-27b-it

## Step 3: Set Your Token (PowerShell)

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

## Step 4: Run the Test

```powershell
python test_medgemma.py
```

## What Will Happen?

The script will:
1. ✅ Verify your authentication
2. 📥 Download the MedGemma 4B model (~8.5GB - first time only)
3. 🧪 Test with 3 sample medical queries:
   - "What are the common symptoms of type 2 diabetes?"
   - "Explain the mechanism of action of metformin."
   - "What are the differential diagnoses for chest pain?"
4. 💡 Display the AI's medical responses

## Expected First Run

⏱️ **First time**: 15-30 minutes (downloading model)
⏱️ **Subsequent runs**: 1-2 minutes (model cached)

## System Requirements

- **Storage**: 10GB free space
- **RAM**: 16GB recommended (8GB minimum)
- **GPU**: Optional but recommended for faster inference

## Troubleshooting

### "401 Client Error" or "403 Forbidden"
- Make sure you accepted the model terms (Step 2)
- Verify your token is set correctly
- Check token has **read** permissions

### Out of Memory
- Close other applications
- The model will use CPU if GPU is unavailable (slower but works)

### Slow Performance
- Normal on first run (downloading model)
- CPU inference is slower than GPU
- Consider using GPU if available

## Next Steps

Once working, you can:
- Modify queries in `test_medgemma.py`
- Try the larger 27B model (requires more RAM)
- Experiment with generation parameters
- Use for your specific medical research questions

---

**⚠️ Important**: MedGemma is for research/educational purposes only. Not for medical diagnosis or treatment.
