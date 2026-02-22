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
    "current_model_name": None,
    "det_processor": None,
    "det_model": None
}

class ChatRequest(BaseModel):
    prompt: str
    model_name: str
    hf_token: str
    analysis_mode: Optional[str] = "General Analysis"
    pdf_text: Optional[str] = None

class ReportRequest(BaseModel):
    prompt: Optional[str] = None
    response: Optional[str] = None
    thinking: Optional[str] = None
    metadata: Optional[dict] = None
    image_b64: Optional[str] = None # Base64 encoded image
    user_comment: Optional[str] = None

# --- Helper Functions (Ported from app.py) ---


def load_detection_service():
    if MODELS["det_model"] is not None:
        return MODELS["det_processor"], MODELS["det_model"]

    print("Loading Grounding DINO for Object Detection (GPU preferred)...")
    from transformers import AutoProcessor as AP, AutoModelForZeroShotObjectDetection
    det_processor = AP.from_pretrained("IDEA-Research/grounding-dino-tiny")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny",
        dtype=torch.float32
    ).to(device)
    
    MODELS["det_processor"] = det_processor
    MODELS["det_model"] = det_model
    return det_processor, det_model

def run_detection(image, targets_text, threshold):
    det_processor, det_model = load_detection_service()
    device = det_model.device
    
    text_prompt = targets_text.strip().rstrip(".") + "."
    inputs = det_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    print(f"Running detection for: {text_prompt}")
    with torch.no_grad():
        outputs = det_model(**inputs)
    
    results = det_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]] # (height, width)
    )[0]
    
    print(f"Detection results: Found {len(results['boxes'])} regions.")
    for score, label in zip(results["scores"], results["labels"]):
        print(f" - Found {label}: {score:.2f}")
        
    return results

def load_medgemma_service(name: str, token: str):
    if MODELS["current_model_name"] == name and MODELS["model"] is not None:
        return MODELS["tokenizer"], MODELS["processor"], MODELS["model"]

    print(f"Loading model: {name}...")
    
    # Handle GGUF Models
    if "GGUF" in name:
        from huggingface_hub import hf_hub_download
        print(f"Detected GGUF model. Downloading quantized weights...")
        
        # Unsloth GGUF filenames can vary (e.g., medgemma-3-4b-it-UD-Q8_K_XL.gguf)
        # We'll try to find any .gguf file that isn't a projector (mmproj)
        from huggingface_hub import list_repo_files
        try:
            files = list_repo_files(name, token=token)
            gguf_files = [f for f in files if f.endswith(".gguf") and "mmproj" not in f]
            if not gguf_files:
                raise Exception("No GGUF file found in repo")
            gguf_filename = gguf_files[0]
            print(f"Found GGUF file: {gguf_filename}")
            model_path = hf_hub_download(repo_id=name, filename=gguf_filename, token=token)
        except Exception as e:
            print(f"Dynamic search failed: {e}. Trying fallback...")
            gguf_filename = "medgemma-4b-it-Q4_K_M.gguf"
            try:
                model_path = hf_hub_download(repo_id=name, filename=gguf_filename, token=token)
            except:
                raise HTTPException(status_code=500, detail=f"GGUF file not found. Checked: {name}")

        tokenizer = AutoTokenizer.from_pretrained(name, token=token)
        try:
            # For vision support, we often need the processor from the base model 
            # if not fully included in the GGUF repo
            processor = AutoProcessor.from_pretrained(name, token=token, use_fast=False)
        except:
            # Fallback to the non-GGUF version for processor if needed
            base_model = name.replace("-GGUF", "")
            processor = AutoProcessor.from_pretrained(base_model, token=token, use_fast=False)

        model = AutoModelForCausalLM.from_pretrained(
            name,
            gguf_file=model_path,
            token=token,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, token=token)
        try:
            processor = AutoProcessor.from_pretrained(name, token=token, use_fast=False)
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
            dtype=torch.bfloat16,
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
        
        pixel_array = ds.pixel_array
        
        # 1. Handle VOI LUT / Windowing
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            pixel_array = apply_voi_lut(pixel_array, ds)
        elif hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # 2. Robust Shape Handling
        pixel_array = np.squeeze(pixel_array) # Remove (1, 512, 512) -> (512, 512)
        
        if pixel_array.ndim == 3:
            # If it's a volume (Slices, H, W), pick the middle slice
            pixel_array = pixel_array[pixel_array.shape[0] // 2]
        elif pixel_array.ndim == 1:
            # If it's a 1D signal, reshape to a square if possible or a thin strip
            side = int(len(pixel_array)**0.5)
            if side * side == len(pixel_array):
                pixel_array = pixel_array.reshape((side, side))
            else:
                pixel_array = pixel_array.reshape((1, -1))

        # 3. Normalize to uint8
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max > p_min:
            pixel_array = ((pixel_array - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            
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
                "You are an expert radiologist. Analyze the image and identify findings.\n"
                "CRITICAL: For every finding, you MUST provide coordinates in the exact format:\n"
                "FINDING: <Name>\n"
                "LOCATION: [y_min, x_min, y_max, x_max]\n"
                "DESCRIPTION: <Brief explanation>\n\n"
                "Use normalized coordinates (0.0 to 1.0). "
                "The UI will use these to draw boxes directly on the image.\n"
            ),
            "Radiology Report": (
                "You are an expert radiologist. Write a detailed, structured radiology report.\n"
                "Include these sections:\n"
                "CLINICAL INDICATION:\nTECHNIQUE:\nFINDINGS:\nIMPRESSION:\nRECOMMENDATIONS:\n"
            ),
            "Patient Consultation": (
                "You are a compassionate doctor conducting a patient intake.\n"
                "STRICT RULES:\n"
                "1. Ask EXACTLY ONE short, leading question at a time.\n"
                "2. DO NOT provide summaries, diagnosis, or long explanations yet.\n"
                "3. Focus on building the patient's history (Chief Complaint -> HPI -> Symptoms -> History).\n"
                "4. Be empathetic but move the consultation forward with your next question.\n"
                "5. Only when you have a full picture (after 5-8 turns), provide a 'Clinical Impression'.\n\n"
                "Begin by introducing yourself and asking why they are seeking help today."
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

        # ── Pre-process: Object Detection (Grounding DINO) ────────────────────
        detection_results_text = ""
        if img and analysis_mode == "Localization":
            try:
                # Default targets for medical imaging
                targets = "tumor, nodule, mass, lesion, fracture, opacity, effusion"
                results = run_detection(img, targets, 0.3)
                
                if len(results["boxes"]) > 0:
                    detection_results_text = "\n[Initial Detection Context]:\n"
                    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                        # Convert absolute pixels to normalized 0-1 for model context
                        w, h = img.size
                        x0, y0, x1, y1 = box.tolist() # DINO returns [xmin, ymin, xmax, ymax]
                        norm_box = [round(y0/h, 3), round(x0/w, 3), round(y1/h, 3), round(x1/w, 3)]
                        detection_results_text += f"FINDING: {label.upper()}\nLOCATION: {norm_box}\n"
                    
                    # Inject detection results into the prompt to "help" the model
                    msgs[-1]["content"] += f"\n\nPre-detection hints (verify and refine these in your response):\n{detection_results_text}"
            except Exception as e:
                print(f"Detection error (skipping): {e}")

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

        # Handle Gemma 3 thinking tokens (<unused95> ... <unused94>)
        gemma3_think = re.search(r"<unused95>(.*?)<unused94>(.*)", full_output, re.DOTALL | re.IGNORECASE)
        if gemma3_think:
            thinking_text = gemma3_think.group(1).strip()
            response_text = gemma3_think.group(2).strip()
        else:
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

        # Final cleanup for any stray tokens in the response
        response_text = re.sub(r"<(/?)(thinking|unused94|unused95|thought|thought_process|thinking_process)>", "", response_text, flags=re.IGNORECASE).strip()
        # Also clean up common markdown-style headers that the model might use for thinking
        response_text = re.sub(r"^(###\s+Thinking|Thinking:|Thought:)\s*", "", response_text, flags=re.IGNORECASE | re.MULTILINE).strip()

        return {
            "response": response_text,
            "thinking": thinking_text,  # Empty string if model didn't output thinking
            "full_output": full_output   # Raw output for debugging
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report")
async def report(req: ReportRequest):
    """
    Receives a reported case from the user.
    In a real-world scenario, you would send this to your email, 
    a Discord webhook, or a database.
    """
    try:
        # 1. Log to local Colab console (developer can see it in logs)
        print("\n" + "="*50)
        print("🚩 NEW CASE REPORTED")
        print(f"Prompt: {req.prompt[:100]}...")
        print(f"Comment: {req.user_comment}")
        print("="*50 + "\n")

        # 2. Save locally in Colab for the developer to download later
        report_dir = "Medgemma_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        import time
        report_id = int(time.time())
        report_path = os.path.join(report_dir, f"report_{report_id}.json")
        
        with open(report_path, "w") as f:
            f.write(req.json(indent=2))

        # 3. (Optional) Forward to a central Webhook if configured
        webhook_url = os.getenv("REPORT_WEBHOOK_URL")
        if webhook_url:
            import httpx
            async with httpx.AsyncClient() as client:
                # We send a truncated version to Discord for speed
                payload = {
                    "content": f"🩺 **New MedGemma Report**\n**Comment:** {req.user_comment}\n**Prompt:** {req.prompt[:500]}",
                    "username": "MedGemma Reporter"
                }
                await client.post(webhook_url, json=payload)

        return {"status": "success", "message": "Report received. Thank you for your feedback!"}
    except Exception as e:
        print(f"Report error: {e}")
        return {"status": "partial_success", "message": "Report saved locally but failed to send to webhook."}


if __name__ == "__main__":
    import threading
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000),
        daemon=True
    )
    server_thread.start()
    server_thread.join()
