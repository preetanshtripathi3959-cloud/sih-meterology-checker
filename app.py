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
st.set_page_config(page_title="SIH: Legal Metrology AI", layout="wide")

st.markdown("""
    <style>
    .report-card { background: white; padding: 15px; border-radius: 10px; border-left: 10px solid #004085; color: black; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .status-pass { color: #28a745 !important; font-weight: bold; }
    .status-fail { color: #dc3545 !important; font-weight: bold; }
    .data-box { background: #e9ecef; padding: 5px 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; margin-top: 5px; display: inline-block; }
    </style>
""", unsafe_allow_html=True)

# --- MODELS ---
@st.cache_resource
def load_ai():
    model = YOLO('best.pt') if os.path.exists('best.pt') else YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'], gpu=False) 
    return model, reader

detector, reader = load_ai()

# --- OPTIMIZED FAST PROCESSING ---
def fast_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 1.5x (Faster than 3x, but keeps detail)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return gray

# --- COMPLIANCE LOGIC WITH RESULT EXTRACTION ---
def run_compliance_analysis(extracted_list):
    full_blob = " ".join(extracted_list).lower()
    
    # 1. Price Extraction
    mrp_match = re.search(r"(?:mrp|rs|price|₹)\.?\s?(\d+(?:\.\d+)?)", full_blob)
    tax_score = fuzz.partial_ratio("inclusive of all taxes", full_blob)
    mrp_pass = (mrp_match is not None) and (tax_score > 50)

    # 2. Quantity Extraction
    qty_match = re.search(r"(\d+\.?\d*)\s?(ml|g|kg|l|unit|n|pcs|gm)", full_blob)
    
    # 3. Date Extraction
    date_match = re.search(r"(\d{2}[/\-\.]\d{2,4})", full_blob)

    # Compile Results
    report = {
        "MRP & Taxes": (bool(mrp_pass), f"Price: {mrp_match.group(0) if mrp_match else 'Not Found'}"),
        "Net Quantity": (bool(qty_match), f"Qty: {qty_match.group(0) if qty_match else 'Not Found'}"),
        "Mfg Date": (bool(date_match), f"Date: {date_match.group(0) if date_match else 'Not Found'}")
    }
    return report, full_blob

# --- MAIN UI ---
st.title("⚖️ Legal Metrology: Smart Inspector")
st.write("Fast Automated Compliance for Packaged Commodities.")

img_file = st.camera_input("Scan Label")

if img_file:
    # 1. LOAD & RESIZE (Speed Booster)
    raw_image = ImageOps.exif_transpose(Image.open(img_file))
    img_np = np.array(raw_image)
    h, w = img_np.shape[:2]
    # Resize input to 800px width (Perfect balance of speed/accuracy)
    img_np = cv2.resize(img_np, (800, int(h * 800 / w)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing Components..."):
        # 2. YOLO Inference
        results = detector(img_bgr, conf=0.10)
        detected_texts = []
        
        # 3. Smart OCR Logic
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(0, x1-15), max(0, y1-15), min(img_bgr.shape[1], x2+15), min(img_bgr.shape[0], y2+15)
                crop = img_bgr[y1:y2, x1:x2]
                # Paragraph mode grouping for speed
                txt = reader.readtext(fast_enhance(crop), detail=0, paragraph=True)
                detected_texts.extend(txt)
        
        # Always run one fast full-page scan as backup
        detected_texts.extend(reader.readtext(fast_enhance(img_bgr), detail=0, paragraph=True))

        # 4. RESULTS ENGINE
        report, raw_text = run_compliance_analysis(detected_texts)

        # 5. UI DISPLAY
        col_vis, col_rep = st.columns(2)
        
        with col_vis:
            st.subheader("AI Vision")
            # YOLO BGR to RGB
            annotated_frame = results[0].plot()
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col_rep:
            st.subheader("Compliance Report")
            for rule, (status, data_found) in report.items():
                s_icon = "✅ PASS" if status else "❌ FAIL"
                s_class = "status-pass" if status else "status-fail"
                
                st.markdown(f"""
                    <div class="report-card">
                        <b>{rule}</b>: <span class="{s_class}">{s_icon}</span><br>
                        <div class="data-box">{data_found}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show the raw text found for judges
            with st.expander("Show Extracted Raw Text"):
                st.write(raw_text)
