import torch

# FIX: Allow YOLOv8 classes to load in newer PyTorch versions
try:
    from ultralytics.nn.tasks import DetectionModel
    # This tells PyTorch that the YOLO model structure is safe to load
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import re
from rapidfuzz import fuzz

# --- MEMORY-OPTIMIZED MODEL LOADING ---
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    model_path = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
    return YOLO(model_path)

@st.cache_resource
def load_ocr():
    import easyocr
    # gpu=False is CRITICAL to prevent 'Oh no' memory crashes on Streamlit Cloud
    return easyocr.Reader(['en'], gpu=False)

# --- UI CONFIGURATION ---
st.set_page_config(page_title="SIH: Legal Metrology AI", layout="wide")

# Custom CSS for high visibility in Dark/Light modes
st.markdown("""
    <style>
    .report-card { 
        background: #ffffff; padding: 15px; border-radius: 10px; 
        border-left: 8px solid #004085; color: #111111 !important;
        margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
    }
    .status-pass { color: #1e7e34 !important; font-weight: bold; }
    .status-fail { color: #bd2130 !important; font-weight: bold; }
    .card-title { color: #004085 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- IMAGE ENHANCEMENT ENGINE ---
def enhance_for_ocr(img_crop):
    # 1. Gray & Zoom (2x scale) - Best for small labels
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # 2. Sharpening (Unsharp Mask)
    gaussian = cv2.GaussianBlur(gray, (0,0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
    return sharpened

# --- COMPLIANCE ENGINE ---
def check_compliance(text_list, lang_txt):
    full_blob = " ".join(text_list).lower()
    
    # Fuzzy Matching for "Inclusive of all taxes" (Rule 6)
    tax_similarity = fuzz.partial_ratio("inclusive of all taxes", full_blob)
    mrp_found = re.search(r"(mrp|rs|retail|price).?\s?\d+", full_blob)
    
    # Net Quantity (Rule 7)
    qty_found = re.search(r"(\d+\.?\d*)\s?(g|kg|ml|l|unit|n|pcs|gm)", full_blob)
    
    # Mfg Date (Rule 9)
    date_found = re.search(r"(\d{2}[/\-\.]\d{2,4})", full_blob) or "mfg" in full_blob or "pkd" in full_blob

    return {
        lang_txt["mrp"]: (tax_similarity > 65 or mrp_found, "Req: MRP + 'Inclusive of all taxes'"),
        lang_txt["qty"]: (bool(qty_found), "Req: Net Weight/Volume (e.g. 50ml)"),
        lang_txt["date"]: (bool(date_found), "Req: Month/Year of packing")
    }

# --- MAIN APP UI ---
st.title("🛡️ Legal Metrology Compliance AI")

# Sidebar Diagnostics
with st.sidebar:
    st.header("🛠️ AI Debugger")
    lang_choice = st.selectbox("Language / भाषा", ["English", "Hindi (हिन्दी)"])
    conf_val = st.slider("AI Confidence", 0.01, 1.0, 0.15)
    st.divider()
    try:
        detector = load_yolo()
        reader = load_ocr()
        st.success("AI Models Active")
        st.write(f"Classes: `{detector.names}`")
    except Exception as e:
        st.error(f"Load Error: {e}")

# Translation Data
T = {
    "English": {"mrp": "MRP & Taxes", "qty": "Net Quantity", "date": "Mfg Date", "scan": "Scan Label", "rep": "Report"},
    "Hindi (हिन्दी)": {"mrp": "MRP और कर", "qty": "शुद्ध मात्रा", "date": "निर्माण तिथि", "scan": "लेबल स्कैन करें", "rep": "रिपोर्ट"}
}
L = T[lang_choice]

img_file = st.camera_input(L["scan"])

if img_file:
    # 1. Load & Resize to save Memory (OOM protection)
    image = Image.open(img_file)
    image = ImageOps.exif_transpose(image)
    img_np = np.array(image)
    
    h, w = img_np.shape[:2]
    if w > 1200: # Resize if phone photo is too large
        img_np = cv2.resize(img_np, (1200, int(h * (1200 / w))))
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing Compliance..."):
        # 2. YOLO Detection
        results = detector(img_bgr, conf=conf_val)
        detected_texts = []

        if len(results[0].boxes) > 0:
            st.sidebar.write(f"Found {len(results[0].boxes)} zones")
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add Padding
                x1, y1 = max(0, x1-15), max(0, y1-15)
                x2, y2 = min(img_bgr.shape[1], x2+15), min(img_bgr.shape[0], y2+15)
                
                crop = img_bgr[y1:y2, x1:x2]
                enhanced = enhance_for_ocr(crop)
                
                txt = reader.readtext(enhanced, detail=0)
                detected_texts.extend(txt)
        
        # 3. Always Full-Page Fallback
        full_enhanced = enhance_for_ocr(img_bgr)
        detected_texts.extend(reader.readtext(full_enhanced, detail=0))
        
        # 4. Show Results
        report = check_compliance(list(set(detected_texts)), L)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("AI Vision")
            # FIX: BGR to RGB
            annotated = results[0].plot()
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.subheader(L["rep"])
            for rule, (status, desc) in report.items():
                s_icon = "✅" if status else "❌"
                s_class = "status-pass" if status else "status-fail"
                st.markdown(f"""
                    <div class="report-card">
                        <span class="card-title">{rule}</span>: 
                        <span class="{s_class}">{s_icon}</span><br>
                        <small>{desc}</small>
                    </div>
                """, unsafe_allow_html=True)
            
            with st.expander("Show AI Raw Data"):
                st.write(detected_texts)
