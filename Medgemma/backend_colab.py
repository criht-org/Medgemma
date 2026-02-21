import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image, ImageDraw
import io
import numpy as np
import pydicom
try:
    from pydicom.pixels import apply_voi_lut  # pydicom >= 3.0
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_voi_lut  # pydicom < 3.0
from pypdf import PdfReader

app = FastAPI(title="MedGemma Colab Backend")

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State for Models ---
MODELS = {
    "tokenizer": None,
    "processor": None,
    "model": None,
    "current_model_name": None
}

class ChatRequest(BaseModel):
    prompt: str
    model_name: str
    hf_token: str
    analysis_mode: Optional[str] = "General Analysis"
    pdf_text: Optional[str] = None

# --- Helper Functions (Ported from app.py) ---

def load_medgemma_service(name: str, token: str):
    if MODELS["current_model_name"] == name and MODELS["model"] is not None:
        return MODELS["tokenizer"], MODELS["processor"], MODELS["model"]

    print(f"Loading model: {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name, token=token)
    try:
        processor = AutoProcessor.from_pretrained(name, token=token)
    except:
        processor = None

    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    ) if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=q,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    MODELS["tokenizer"] = tokenizer
    MODELS["processor"] = processor
    MODELS["model"] = model
    MODELS["current_model_name"] = name
    
    return tokenizer, processor, model

def process_dicom_bytes(file_bytes):
    try:
        with io.BytesIO(file_bytes) as f:
            ds = pydicom.dcmread(f)
        
        pixel_array = apply_voi_lut(ds.pixel_array, ds) if hasattr(ds, 'WindowCenter') else ds.pixel_array
        if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-7) * 255).astype(np.uint8)
        img = Image.fromarray(pixel_array).convert("RGB")
        
        meta = {
            "Modality": str(getattr(ds, 'Modality', 'Unknown')),
            "Patient": str(getattr(ds, 'PatientName', 'Anonymous')),
            "Study": str(getattr(ds, 'StudyDescription', 'N/A')),
            "Date": str(getattr(ds, 'ContentDate', 'N/A'))
        }
        return img, meta
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DICOM processing error: {str(e)}")

# --- Endpoints ---

@app.get("/")
async def root():
    # If the built frontend exists, serve it
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dist = os.path.join(base_dir, "frontend", "dist")
    index_path = os.path.join(frontend_dist, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    # Fallback to status message
    return {
        "status": "online", 
        "message": "MedGemma Colab Backend is running",
        "hint": "Build the frontend to see the UI: cd Medgemma/frontend && npm install && npm run build"
    }

# Mount static assets if build exists
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_assets = os.path.join(base_dir, "frontend", "dist", "assets")
if os.path.exists(frontend_assets):
    app.mount("/assets", StaticFiles(directory=frontend_assets), name="assets")

@app.post("/process-upload")
async def process_upload(file: UploadFile = File(...)):
    contents = await file.read()
    ext = file.filename.split(".")[-1].lower()
    
    if ext == "dcm" or file.content_type == "application/dicom":
        img, meta = process_dicom_bytes(contents)
        # We can't easily send the image back in a JSON without encoding to base64
        # For simplicity, we'll return the meta and expect the frontend to handle the local preview
        return {"type": "dicom", "metadata": meta}
    
    elif file.content_type.startswith("image/"):
        return {"type": "image", "filename": file.filename}
    
    elif ext == "pdf":
        with io.BytesIO(contents) as f:
            reader = PdfReader(f)
            text = " ".join([p.extract_text() for p in reader.pages])
        return {"type": "pdf", "text": text[:2000]} # Limit text length

    return {"type": "unknown", "filename": file.filename}

import json as _json

@app.post("/chat")
async def chat(
    prompt: str = Form(...),
    model_name: str = Form(...),
    hf_token: str = Form(...),
    analysis_mode: str = Form("General Analysis"),
    history: str = Form("[]"),           # JSON array of {role, content} dicts
    file: Optional[UploadFile] = File(None)
):
    try:
        tok, proc, model = load_medgemma_service(model_name, hf_token)

        # Parse conversation history sent from frontend
        try:
            conv_history = _json.loads(history)   # [{role, content}, ...]
        except Exception:
            conv_history = []

        # ── Mode-specific system prompts ──────────────────────────────────────
        is_consultation = analysis_mode == "Patient Consultation"

        mode_prompts = {
            "Localization": (
                "You are an expert radiologist. Carefully examine the provided medical image.\n"
                "For each significant finding, provide:\n"
                "1. Finding name\n"
                "2. Normalized bounding box coordinates as [y_min, x_min, y_max, x_max] "
                "(values between 0.0 and 1.0 relative to image dimensions)\n"
                "3. Brief description\n\n"
                "Format each finding as:\n"
                "FINDING: <name>\n"
                "LOCATION: [y_min, x_min, y_max, x_max]\n"
                "DESCRIPTION: <brief description>\n"
            ),
            "Radiology Report": (
                "You are an expert radiologist. Write a detailed, structured radiology report.\n"
                "Include these sections:\n"
                "CLINICAL INDICATION:\nTECHNIQUE:\nFINDINGS:\nIMPRESSION:\nRECOMMENDATIONS:\n"
            ),
            "Patient Consultation": (
                "You are a compassionate and highly experienced clinician conducting a structured "
                "patient intake consultation.\n\n"
                "Your approach:\n"
                "- Greet the patient warmly and professionally.\n"
                "- Ask ONE focused, open-ended leading question at a time — never ask multiple questions at once.\n"
                "- Systematically gather clinical history using a logical sequence:\n"
                "  Chief Complaint → History of Present Illness → Duration & Onset → "
                "  Severity (1-10 scale) → Associated Symptoms → Aggravating/Relieving Factors → "
                "  Past Medical History → Medications → Family History → Social History.\n"
                "- Acknowledge the patient's response empathetically before asking the next question.\n"
                "- Use simple, plain language a patient can understand (avoid medical jargon).\n"
                "- When you have gathered sufficient history, end with a brief clinical summary "
                "and possible next steps.\n\n"
                "Begin the consultation now."
            ),
        }

        system_prompt = mode_prompts.get(analysis_mode, "You are an expert medical AI assistant.")

        # ── Build multi-turn message list ────────────────────────────────────
        # For consultation mode, the system prompt is the FIRST user turn trigger.
        # For other modes, it prefixes the current user message.
        if is_consultation:
            # Build full conversation from history
            msgs = []
            # Seed with system context on first turn only
            if not conv_history:
                msgs.append({"role": "user", "content": system_prompt})
                msgs.append({"role": "model", "content": "Hello! I'm Dr. AI. I'm here to help understand your symptoms. Could you please start by telling me what brings you in today? What is your main concern?"})
            # Append all previous turns
            for turn in conv_history:
                msgs.append({"role": turn["role"] if turn["role"] != "assistant" else "model", "content": turn["content"]})
            # Append current user message
            msgs.append({"role": "user", "content": prompt})
        else:
            full_p = f"{system_prompt}\n\nClinical Query: {prompt}"
            msgs = []
            # Inject previous turns for context (last 6 turns max to save tokens)
            for turn in conv_history[-6:]:
                msgs.append({"role": turn["role"] if turn["role"] != "assistant" else "model", "content": turn["content"]})
            msgs.append({"role": "user", "content": full_p})

        # ── Load image if provided ──────────────────────────────────────────
        img = None
        if file:
            raw = await file.read()
            if file.content_type and file.content_type.startswith("image/"):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            elif file.filename and (file.filename.lower().endswith(".dcm") or file.content_type == "application/dicom"):
                img, _ = process_dicom_bytes(raw)

        # ── Build model inputs ────────────────────────────────────────────────
        if img and proc:
            # Image mode: inject image into the last user message
            last_user = msgs[-1]
            msgs[-1] = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": last_user["content"]}]}
            input_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=input_text, images=img, return_tensors="pt").to(model.device)
        else:
            input_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = tok(input_text, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # ── Generate ─────────────────────────────────────────────────────────
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=2048,      # Increased from 500
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,   # Prevent repetition loops
            )

        # ── Decode only the NEW tokens (not the input) ────────────────────
        new_tokens = gen_ids[:, input_len:]
        full_output = tok.decode(new_tokens[0], skip_special_tokens=True).strip()

        # ── Separate <thinking> from final answer ─────────────────────────
        import re
        thinking_text = ""
        response_text = full_output

        # Handle explicit <thinking>...</thinking> blocks
        think_match = re.search(r"<thinking>(.*?)</thinking>(.*)", full_output, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_text = think_match.group(1).strip()
            response_text = think_match.group(2).strip()
        else:
            # Some models use "Thinking:" prefix without tags
            think_prefix = re.match(r"^(Thinking:.*?\n\n)(.*)", full_output, re.DOTALL | re.IGNORECASE)
            if think_prefix:
                thinking_text = think_prefix.group(1).strip()
                response_text = think_prefix.group(2).strip()

        return {
            "response": response_text,
            "thinking": thinking_text,  # Empty string if model didn't output thinking
            "full_output": full_output   # Raw output for debugging
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import threading
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000),
        daemon=True
    )
    server_thread.start()
    server_thread.join()
