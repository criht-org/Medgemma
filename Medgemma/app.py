import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    DICOM_SUPPORTED = True
except ImportError:
    DICOM_SUPPORTED = False

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="INNOVDOC AI", page_icon="✨", layout="wide")

# ── UI Strategy: Gemini Aesthetic ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700&display=swap');

    :root {
        --gemini-bg: #ffffff;
        --gemini-user-chat: #e9eef6;
        --gemini-accent: #1a73e8;
        --gemini-text: #1f1f1f;
        --gemini-text-muted: #444746;
        --shadow: 0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15);
    }

    .stApp {
        background-color: var(--gemini-bg);
        font-family: 'Inter', sans-serif;
    }

    /* Centered Layout */
    .block-container {
        max-width: 850px !important;
        padding-top: 2rem !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f0f4f8;
        border-right: none;
    }

    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        text-align: center;
        color: var(--gemini-text-muted);
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Chat Bubbles */
    .stChatMessage {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 2rem !important;
    }

    /* User Message Bubble */
    [data-testid="chat-message-user"] {
        flex-direction: row-reverse;
    }
    
    [data-testid="chat-message-user"] .stChatMessageContent {
        background-color: var(--gemini-user-chat) !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 12px 20px !important;
        color: var(--gemini-text) !important;
        box-shadow: none !important;
    }

    /* Assistant Message Bubble */
    [data-testid="chat-message-assistant"] .stChatMessageContent {
        background-color: transparent !important;
        padding: 12px 0 !important;
        color: var(--gemini-text) !important;
    }

    /* Input Bar: Pill-shaped like Gemini */
    .stChatInputContainer {
        border-radius: 30px !important;
        border: 1px solid #c4c7c5 !important;
        padding: 5px 15px !important;
        background: #f8fafd !important;
        transition: border 0.2s;
        bottom: 30px !important;
    }
    
    .stChatInputContainer:focus-within {
        border: 1px solid var(--gemini-accent) !important;
        box-shadow: 0 0 0 1px var(--gemini-accent) !important;
    }

    /* Metadata Card */
    .patient-card {
        background: #fdfdfd;
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid #e3e3e3;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }

    .patient-card-header {
        color: var(--gemini-accent);
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* Image Styling */
    img {
        border-radius: 12px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family: Outfit; color: #1a73e8;'>Settings</h2>", unsafe_allow_html=True)
    st.divider()

    hf_token = st.text_input("Hugging Face API Token", type="password", value=os.getenv("HF_TOKEN", ""))
    model_choice = st.selectbox("Model", 
                                ["google/medgemma-1.5-4b-it", "google/medgemma-4b-it", "google/medgemma-27b-it"])
    
    st.divider()
    st.subheader("Capabilities")
    detection_enabled = st.toggle("Object Detection (Grounding DINO)", value=False)
    det_targets = st.text_input("Detection Targets", value="tumor, lesion", disabled=not detection_enabled)
    det_conf = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05, disabled=not detection_enabled)

    st.divider()
    st.subheader("Profiles")
    analysis_mode = st.selectbox("Report Mode", ["General Analysis", "Localization", "Radiology Report"])

    st.divider()
    st.caption("✨ Developed by Dr. R. K. Ramanan")

# ── Business Logic ────────────────────────────────────────────────────────────
def process_dicom(file):
    if not DICOM_SUPPORTED:
        return None, "DICOM libraries not installed."
    try:
        ds = pydicom.dcmread(file)
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
        return None, str(e)

@st.cache_resource
def load_medgemma(name, token):
    if not token: return None, None, None
    from transformers import AutoProcessor, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(name, token=token)
    try: processor = AutoProcessor.from_pretrained(name, token=token)
    except: processor = None
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True) if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, token=token, torch_dtype=torch.bfloat16, device_map="auto" if torch.cuda.is_available() else None)
    return tokenizer, processor, model

@st.cache_resource
def load_dino():
    from transformers import AutoProcessor as AP, AutoModelForZeroShotObjectDetection
    return AP.from_pretrained("IDEA-Research/grounding-dino-tiny"), AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny", torch_dtype=torch.float32)

# ── Chat State & Persistance ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_file" not in st.session_state:
    st.session_state.pending_file = None

# Header
st.markdown("<div class='main-header'>INNOVDOC AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Expert medical insights with✨Gemini Aesthetic</div>", unsafe_allow_html=True)

# ── Integrated-style Upload (Gemini Style) ────────────────────────────────────
# We place it in the sidebar or a fixed small area to avoid clunky UI
with st.container():
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        # A tiny icon-based uploader
        up_file = st.file_uploader("📁", type=["png", "jpg", "dcm", "pdf"], label_visibility="collapsed")
        if up_file:
            st.session_state.pending_file = up_file

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Persistently display images and metadata from the message state
        if "img" in msg and msg["img"] is not None:
            st.image(msg["img"], width=450, caption="Attached Scan")
        if "meta" in msg and msg["meta"] is not None:
            m = msg["meta"]
            st.markdown(f"""
            <div class="patient-card">
                <div class="patient-card-header">Clinical Record Summary</div>
                <b>Patient ID:</b> {m['Patient']}<br>
                <b>Modality:</b> {m['Modality']} | <b>Study:</b> {m['Study']}
            </div>
            """, unsafe_allow_html=True)
        st.markdown(msg["content"])

# ── Chat Input Logic ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a clinical question or describe the scan..."):
    # Build current state for the message
    curr_msg = {"role": "user", "content": prompt, "img": None, "meta": None}
    
    # Process pending file if it exists
    if st.session_state.pending_file:
        f = st.session_state.pending_file
        ext = f.name.split(".")[-1].lower()
        if ext == "dcm" or f.type == "application/dicom":
            img, meta = process_dicom(f)
            curr_msg["img"] = img
            curr_msg["meta"] = meta
        elif f.type.startswith("image/"):
            curr_msg["img"] = Image.open(f).convert("RGB")
        elif ext == "pdf":
            reader = PdfReader(f)
            curr_msg["pdf_data"] = " ".join([p.extract_text() for p in reader.pages])
        
        # Clear pending file once processed for this turn
        st.session_state.pending_file = None

    # Save and Render User Message
    st.session_state.messages.append(curr_msg)
    with st.chat_message("user"):
        if curr_msg["img"]:
            st.image(curr_msg["img"], width=450)
        st.markdown(prompt)

    # Assistant Turn
    with st.chat_message("assistant"):
        if not hf_token:
            st.info("💡 Please provide your Hugging Face API Token in the sidebar.")
        else:
            with st.spinner("✨"):
                # 1. Image Detection
                if detection_enabled and curr_msg["img"]:
                    p_dino, m_dino = load_dino()
                    text_p = det_targets.strip().rstrip(".") + "."
                    inputs = p_dino(images=curr_msg["img"], text=text_p, return_tensors="pt")
                    with torch.no_grad():
                        outs = m_dino(**inputs)
                    results = p_dino.post_process_grounded_object_detection(
                        outs, inputs["input_ids"], threshold=det_conf, text_threshold=det_conf, target_sizes=[curr_msg["img"].size[::-1]]
                    )[0]
                    
                    if len(results["boxes"]) > 0:
                        ann_img = curr_msg["img"].copy()
                        d = ImageDraw.Draw(ann_img)
                        for b, s, l in zip(results["boxes"], results["scores"], results["labels"]):
                            d.rectangle(b.tolist(), outline="#1a73e8", width=4)
                            d.text((b[0], b[1]-12), f"{l} {s:.0%}", fill="#1a73e8")
                        st.image(ann_img, width=450, caption="Detections")

                # 2. MedGemma Reasoning
                tok, proc, model = load_medgemma(model_choice, hf_token)
                if model:
                    p_temp = {
                        "Localization": "Identify findings and report normalized [y0, x0, y1, x1] coords.\nFinding: <name>, Loc: [...]",
                        "Radiology Report": "Write a professional radiology report with Findings, Impression, and Plan."
                    }.get(analysis_mode, "")
                    
                    full_p = f"{p_temp}\n\nClinical Query: {prompt}"
                    if "pdf_data" in curr_msg:
                        full_p = f"Reference Clinical Text:\n{curr_msg['pdf_data'][:800]}\n\nTask: {full_p}"

                    if curr_msg["img"] and proc:
                        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": full_p}]}]
                        txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                        ins = proc(text=txt, images=curr_msg["img"], return_tensors="pt").to(model.device)
                    else:
                        txt = f"<start_of_turn>user\n{full_p}<end_of_turn>\n<start_of_turn>model\n"
                        ins = tok(txt, return_tensors="pt").to(model.device)
                    
                    gen = model.generate(**ins, max_new_tokens=500, temperature=0.7, do_sample=True)
                    response = proc.batch_decode(gen, skip_special_tokens=True)[0].split("model")[-1].strip()
                    
                    # Store and display assistant response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()
st.caption("AI-generated medical information. Always verify with a qualified clinician.")
