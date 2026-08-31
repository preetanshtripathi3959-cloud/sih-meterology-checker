import torch
try:
    from ultralytics.nn.tasks import DetectionModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageOps
import re
import os
from rapidfuzz import fuzz

# --- UI CONFIG ---
st.set_page_config(page_title="Fast Metrology AI", layout="wide")

st.markdown("""
    <style>
    .report-card { background: white; padding: 15px; border-radius: 10px; border-left: 8px solid #004085; color: black; margin-bottom: 10px; }
    .status-pass { color: #28a745 !important; font-weight: bold; }
    .status-fail { color: #dc3545 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- MODELS ---
@st.cache_resource
def load_ai():
    # Load Nano model (Fastest version)
    model = YOLO('best.pt') if os.path.exists('best.pt') else YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'], gpu=False) # Cloud is CPU only
    return model, reader

detector, reader = load_ai()

# --- OPTIMIZED PROCESSING ---
def fast_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reduced zoom from 3x to 1.5x for speed
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return gray

def check_compliance(extracted_list):
    raw_blob = " ".join(extracted_list).lower()
    mrp_pass = (fuzz.partial_ratio("inclusive of all taxes", raw_blob) > 50) or \
               (any(x in raw_blob for x in ['mrp', 'rs', 'price', '₹']) and re.search(r"\d{2,}", raw_blob))
    qty_pass = re.search(r"\d+\s?(ml|g|kg|l|n|unit|pcs|gm)", raw_blob)
    date_pass = re.search(r"(\d{2}[/\-\.]\d{2,4})", raw_blob) or "mfg" in raw_blob
    
    return {
        "MRP & Taxes": (bool(mrp_pass), "MRP + Inclusive of all taxes"),
        "Net Quantity": (bool(qty_pass), "Weight/Volume (e.g. 50ml)"),
        "Mfg Date": (bool(date_pass), "Month/Year of packing")
    }

# --- MAIN UI ---
st.title("⚖️ Fast Legal Metrology AI")

img_file = st.camera_input("Scan Label")

if img_file:
    # 1. Load and RESIZE immediately to save CPU cycles
    raw_image = ImageOps.exif_transpose(Image.open(img_file))
    img_np = np.array(raw_image)
    h, w = img_np.shape[:2]
    # Resize to 640px width (Standard YOLO size)
    img_np = cv2.resize(img_np, (640, int(h * 640 / w)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Fast Analysis..."):
        # 2. YOLO Inference (Faster on small image)
        results = detector(img_bgr, conf=0.10)
        detected_texts = []
        
        # 3. Smart OCR Logic
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Small 10px padding
                x1, y1, x2, y2 = max(0, x1-10), max(0, y1-10), min(img_bgr.shape[1], x2+10), min(img_bgr.shape[0], y2+10)
                crop = img_bgr[y1:y2, x1:x2]
                # Fast OCR with paragraph mode
                txt = reader.readtext(fast_enhance(crop), detail=0, paragraph=True)
                detected_texts.extend(txt)
        else:
            # Only run full scan if YOLO finds nothing (Saves time)
            detected_texts = reader.readtext(fast_enhance(img_bgr), detail=0, paragraph=True)

        # 4. Results
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            report = check_compliance(detected_texts)
            for rule, (status, desc) in report.items():
                s_icon = "✅ PASS" if status else "❌ FAIL"
                s_class = "status-pass" if status else "status-fail"
                st.markdown(f'<div class="report-card"><b>{rule}</b>: <span class="{s_class}">{s_icon}</span><br><small>{desc}</small></div>', unsafe_allow_html=True)
