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
    DICOM_SUPPORTED = True
except ImportError:
    DICOM_SUPPORTED = False

# ── Branding and Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="INNOVDOC -by-doc-for-doc", page_icon="🩺", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .medical-header {
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .doctor-credit {
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Prompt Templates ──────────────────────────────────────────────────────────
PROMPT_TEMPLATES = {
    "General Analysis": None,  # No wrapping — use user's prompt as-is
    "Localization": (
        "You are a medical imaging expert. Analyze this medical image carefully.\n"
        "Identify ALL abnormalities, lesions, or notable findings.\n"
        "For EACH finding you identify, provide:\n"
        "1. Name/description of the finding\n"
        "2. Bounding box coordinates as [y0, x0, y1, x1] where values are normalized to a 0-1000 scale\n"
        "3. Your confidence level (high/medium/low)\n\n"
        "Format each finding as:\n"
        "Finding: <name>\n"
        "Location: [y0, x0, y1, x1]\n"
        "Confidence: <level>\n\n"
        "Then answer the user's question:\n{user_question}"
    ),
    "Detailed Report": (
        "You are a board-certified radiologist. Provide a structured diagnostic report for this medical image.\n\n"
        "Use the following format:\n\n"
        "## FINDINGS\n"
        "For each abnormality, state:\n"
        "- Description and characteristics\n"
        "- Bounding box: [y0, x0, y1, x1] normalized to 0-1000 scale\n"
        "- Severity assessment\n\n"
        "## IMPRESSION\n"
        "Overall diagnostic impression and differential diagnoses.\n\n"
        "## RECOMMENDATION\n"
        "Suggested follow-up studies or clinical actions.\n\n"
        "Clinical question from the referring physician: {user_question}"
    ),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='color: #2e86de;'>INNOVDOC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-style: italic;'>by Dr. R. K. Ramanan</p>", unsafe_allow_html=True)
    st.divider()

    st.header("Settings")
    hf_token = st.text_input("Hugging Face Token", type="password", value=os.getenv("HF_TOKEN", ""))
    model_choice = st.selectbox(
        "Select Model",
        [
            "google/medgemma-1.5-4b-it",
            "google/medgemma-4b-it",
            "google/medgemma-27b-it"
        ],
        help="MedGemma 1.5 supports multimodal analysis (images + text)."
    )

    st.divider()
    st.header("Analysis Mode")
    analysis_mode = st.selectbox(
        "Prompt Template",
        list(PROMPT_TEMPLATES.keys()),
        help="Localization & Detailed Report modes instruct the model to output bounding box coordinates."
    )

    st.divider()
    st.header("🔍 Object Detection")
    detection_enabled = st.toggle("Enable Detection Mode", value=False,
                                   help="Uses Grounding DINO to detect objects in medical images.")
    detection_targets = st.text_input(
        "Detection Targets",
        value="tumor, nodule, fracture, opacity, lesion, mass, effusion",
        help="Comma-separated objects to detect in the image.",
        disabled=not detection_enabled,
    )
    detection_threshold = st.slider(
        "Confidence Threshold", 0.1, 0.9, 0.3, 0.05,
        help="Minimum confidence score for detections.",
        disabled=not detection_enabled,
    )

    st.divider()
    st.header("Medical Files")
    uploaded_file = st.file_uploader("Upload Case File (PDF, Image, DICOM, Text)", type=["pdf", "png", "jpg", "jpeg", "txt", "dcm"])

# ── Persistent State ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# ── Model Loading (Cached) ───────────────────────────────────────────────────
def _get_device_strategy():
    """Auto-detect GPU VRAM and pick the best precision strategy.
    
    - GPU with >=16GB VRAM (e.g., Colab T4/A100): float32, full precision
    - GPU with < 16GB VRAM (e.g., RTX 4060 8GB, A1000 6GB): float16, half precision
    - No GPU: float16 on CPU (saves RAM)
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        if vram_gb >= 16:
            return {
                "torch_dtype": torch.float32,
                "device_map": "auto",
                "label": f"🟢 Full Precision (float32) — {gpu_name} ({vram_gb:.0f} GB)",
            }
        else:
            return {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "label": f"🟡 Half Precision (float16) — {gpu_name} ({vram_gb:.0f} GB)",
            }
    else:
        return {
            "torch_dtype": torch.float16,
            "device_map": None,
            "label": "🔴 CPU mode (float16) — no GPU detected",
        }


@st.cache_resource
def load_model_and_processor(model_name, token):
    if not token:
        return None, None, None
    try:
        from transformers import AutoProcessor
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        try:
            processor = AutoProcessor.from_pretrained(model_name, token=token)
        except:
            processor = None

        strategy = _get_device_strategy()
        st.sidebar.info(strategy["label"])

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=strategy["torch_dtype"],
            device_map=strategy["device_map"],
            token=token
        )
        return tokenizer, processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


@st.cache_resource
def load_detection_model():
    """Load Grounding DINO Tiny for zero-shot object detection (runs on CPU)."""
    try:
        from transformers import AutoProcessor as AP, AutoModelForZeroShotObjectDetection
        det_processor = AP.from_pretrained("IDEA-Research/grounding-dino-tiny")
        det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny",
            torch_dtype=torch.float32,  # CPU — float32 is fine and fast
        )
        det_model.eval()
        return det_processor, det_model
    except Exception as e:
        st.error(f"Could not load Grounding DINO: {e}")
        return None, None


def run_detection(image, targets_text, threshold, det_processor, det_model):
    """Run Grounding DINO on an image and return annotated image + results."""
    # Prepare inputs — Grounding DINO expects a text prompt with "." separators
    text_prompt = targets_text.strip().rstrip(".")  + "."
    inputs = det_processor(images=image, text=text_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = det_model(**inputs)

    # Post-process
    results = det_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    return results


def draw_detection_boxes(image, results):
    """Draw bounding boxes from Grounding DINO results on the image."""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    # Color palette for different detections
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#e84393", "#00cec9", "#fdcb6e"]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    detections_info = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = box.tolist()

        # Draw box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        # Draw label background
        label_text = f"{label} {score:.0%}"
        draw.rectangle([x0, y0 - 20, x0 + len(label_text) * 8, y0], fill=color)
        draw.text((x0 + 4, y0 - 18), label_text, fill="white")

        detections_info.append(f"- **{label}** — confidence: {score:.0%}")

    return draw_img, detections_info

# ── Bounding Box Parsing (improved) ──────────────────────────────────────────
def parse_and_draw_bboxes(response_text, image):
    """Parse bounding box coordinates from model response text and draw them."""
    if not image:
        return None, []

    # Pattern 1: labeled — "Finding: tumor\nLocation: [100, 200, 300, 400]"
    labeled_pattern = r'(?:Finding|Label|Name)[:\s]*([^\n]+)\n\s*(?:Location|Box|Bounding)[:\s]*\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]'
    # Pattern 2: inline labeled — "tumor: [100, 200, 300, 400]"
    inline_pattern = r'(\w[\w\s]{0,30}):\s*\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]'
    # Pattern 3: bare coordinates — [100, 200, 300, 400]
    bare_pattern = r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]'

    bboxes = []

    # Try labeled patterns first
    for match in re.finditer(labeled_pattern, response_text, re.IGNORECASE):
        label = match.group(1).strip()
        coords = [int(match.group(i)) for i in range(2, 6)]
        bboxes.append((label, coords))

    if not bboxes:
        for match in re.finditer(inline_pattern, response_text, re.IGNORECASE):
            label = match.group(1).strip()
            # Skip if label looks like a non-finding word
            if label.lower() in ("location", "box", "bounding", "coordinates", "size", "image"):
                continue
            coords = [int(match.group(i)) for i in range(2, 6)]
            bboxes.append((label, coords))

    if not bboxes:
        for match in re.finditer(bare_pattern, response_text):
            coords = [int(match.group(i)) for i in range(1, 5)]
            # Filter out obviously non-bbox values (e.g. years, small numbers)
            if all(0 <= c <= 1000 for c in coords):
                bboxes.append(("FINDING", coords))

    if not bboxes:
        return None, []

    # Draw boxes
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    width, height = draw_img.size

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#e84393", "#00cec9", "#fdcb6e"]
    findings = []

    for i, (label, coords) in enumerate(bboxes):
        color = colors[i % len(colors)]
        y0, x0, y1, x1 = coords  # MedGemma format: [y0, x0, y1, x1]

        # Convert from 0-1000 normalized to pixel coordinates
        left = x0 * width / 1000
        top = y0 * height / 1000
        right = x1 * width / 1000
        bottom = y1 * height / 1000

        draw.rectangle([left, top, right, bottom], outline=color, width=4)
        # Label background
        draw.rectangle([left, top - 20, left + len(label) * 8 + 8, top], fill=color)
        draw.text((left + 4, top - 18), label.upper(), fill="white")
        findings.append(f"- **{label}** at [{y0}, {x0}, {y1}, {x1}]")

    return draw_img, findings


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("<div class='medical-header'>INNOVDOC</div>", unsafe_allow_html=True)
st.markdown("<div class='doctor-credit'>Powered by MedGemma | Developed by Dr. R. K. Ramanan</div>", unsafe_allow_html=True)

# ── Handle File Content ───────────────────────────────────────────────────────
context = ""
image_data = None
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower() if uploaded_file.name else ""

    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            context += page.extract_text() + "\n"
        st.sidebar.success("PDF Context Loaded!")
    elif file_ext == "dcm" or uploaded_file.type == "application/dicom":
        if not DICOM_SUPPORTED:
            st.sidebar.error("❌ Install pydicom: `pip install pydicom`")
        else:
            ds = pydicom.dcmread(uploaded_file)
            pixel_array = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc = float(ds.WindowCenter[0]) if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter)
                ww = float(ds.WindowWidth[0]) if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth)
                img_min = wc - ww / 2
                img_max = wc + ww / 2
                pixel_array = np.clip(pixel_array, img_min, img_max)
            pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255).astype(np.uint8)
            image_data = Image.fromarray(pixel_array).convert("RGB")
            st.session_state.current_image = image_data
            dicom_info = []
            for tag in ['PatientName', 'Modality', 'StudyDescription', 'BodyPartExamined']:
                if hasattr(ds, tag):
                    dicom_info.append(f"**{tag}**: {getattr(ds, tag)}")
            if dicom_info:
                st.sidebar.info("\n\n".join(dicom_info))
            st.sidebar.success("🏥 DICOM image ready — ask a question to analyze it.")
    elif uploaded_file.type.startswith("image/"):
        image_data = Image.open(uploaded_file).convert("RGB")
        st.session_state.current_image = image_data
        st.sidebar.success("🖼️ Image ready — ask a question to analyze it.")
    else:
        context = uploaded_file.read().decode("utf-8")
        st.sidebar.success("Text File Context Loaded!")

# ── Chat History ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image"):
            st.image(message["image"], caption="Uploaded Medical Image", use_container_width=True)
        st.markdown(message["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("How can I help with your clinical query today?"):
    user_msg = {"role": "user", "content": prompt}
    if image_data:
        user_msg["image"] = image_data
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        if image_data:
            st.image(image_data, caption="Uploaded Medical Image", use_container_width=True)
        st.markdown(prompt)

    if not hf_token:
        st.warning("Please provide your Hugging Face token in the sidebar to start.")
    else:
        # ── Step 1: Grounding DINO Detection (if enabled) ─────────────────
        if detection_enabled and image_data:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Running object detection (Grounding DINO)..."):
                    det_processor, det_model = load_detection_model()
                    if det_processor and det_model:
                        results = run_detection(image_data, detection_targets, detection_threshold, det_processor, det_model)
                        if len(results["boxes"]) > 0:
                            annotated_img, det_info = draw_detection_boxes(image_data, results)
                            st.image(annotated_img, caption="🔍 Grounding DINO Detection Results", use_container_width=True)
                            det_summary = f"**🔍 Detected {len(results['boxes'])} region(s):**\n" + "\n".join(det_info)
                            st.markdown(det_summary)
                            st.session_state.messages.append({"role": "assistant", "content": det_summary})
                        else:
                            st.info("No objects matching the detection targets were found above the confidence threshold.")
                    else:
                        st.warning("Detection model could not be loaded. Continuing with MedGemma analysis only.")

        # ── Step 2: MedGemma Analysis ─────────────────────────────────────
        tokenizer, processor, model = load_model_and_processor(model_choice, hf_token)

        if model:
            with st.chat_message("assistant"):
                with st.spinner("🧠 Analyzing with MedGemma..."):
                    # Apply prompt template
                    template = PROMPT_TEMPLATES[analysis_mode]
                    if template:
                        full_prompt = template.format(user_question=prompt)
                    else:
                        full_prompt = prompt

                    # Add document context if available
                    if context:
                        full_prompt = f"Context:\n{context}\n\n{full_prompt}"

                    # For multimodal models
                    if image_data and processor:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": full_prompt},
                                ],
                            }
                        ]
                        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                        inputs = processor(text=input_text, images=image_data, return_tensors="pt")
                    else:
                        input_text = f"<start_of_turn>user\n{full_prompt}<end_of_turn>\n<start_of_turn>model\n"
                        inputs = tokenizer(input_text, return_tensors="pt")

                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}

                    # Generate response
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True
                    )

                    # Decode
                    if processor and image_data:
                        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                        if "model" in response:
                            response = response.split("model")[-1].strip()
                    else:
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = response.split("<start_of_turn>model\n")[-1]

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # ── Visualize bounding boxes from MedGemma response ───
                    annotated_img, findings = parse_and_draw_bboxes(response, image_data)
                    if annotated_img:
                        st.info("🎯 Medical Localization Detected!")
                        st.image(annotated_img, caption="Grounded Medical Analysis", use_container_width=True)
                        if findings:
                            st.markdown("**Identified Regions:**\n" + "\n".join(findings))
                        st.success("Anatomical regions highlighted based on model findings.")
        else:
            st.error("Model could not be initialized. Check your token and permissions.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Disclaimer: This tool is for research purposes and should not be used as a substitute for professional medical advice.")
