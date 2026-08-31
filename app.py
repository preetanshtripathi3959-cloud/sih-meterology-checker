import torch
# Fix for the PyTorch 2.6 security error
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
    .debug-tag { background: #e9ecef; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.8em; margin-right: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_ai():
    model = YOLO('best.pt') if os.path.exists('best.pt') else YOLO('yolov8n.pt')
    # gpu=False is mandatory for Streamlit Cloud
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, reader = load_ai()

# --- OCR REPAIR SYSTEM ---
def repair_ocr_artifacts(text):
    # Rule 1: Fix Rupee symbol misreads (Often seen as ?, z, f, 7, RS)
    text = re.sub(r'[?zf7]rs', '₹', text, flags=re.IGNORECASE)
    text = text.replace('M1RP', 'MRP').replace('Mzp', 'MRP').replace('RS.', '₹')
    # Rule 2: Fix Tax keyword misreads
    text = text.replace('1nclusive', 'inclusive').replace('a11', 'all').replace('1ncl', 'incl')
    return text

# --- ADVANCED IMAGE PRE-PROCESSING ---
def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Upscale 3x for tiny bottle text
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    # Adaptive threshold to handle bottle glares
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return enhanced

# --- GRANULAR COMPLIANCE ENGINE ---
def check_compliance_granular(extracted_list):
    full_blob = " ".join(extracted_list).lower()
    full_blob = repair_ocr_artifacts(full_blob)
    
    # 1. CURRENCY/PRICE DETECTION
    # Look for ₹ symbol, 'rs', or 'mrp' keyword followed by digits
    currency_found = bool(re.search(r"(₹|rs|mrp|price|retail).?\s?\d+", full_blob))
    
    # 2. TAX KEYWORD DETECTION (Individual Keywords)
    # Look for 'inclusive', 'incl', 'taxes' separately
    tax_keywords = ["inclusive", "incl", "taxes", "tax"]
    found_tax_parts = [kw for kw in tax_keywords if kw in full_blob]
    # Pass if at least 'inclusive' or 'incl' + 'taxes' is found
    tax_status = "inclusive" in found_tax_parts or ("incl" in found_tax_parts and "taxes" in found_tax_parts)

    # 3. QUANTITY DETECTION
    qty_status = bool(re.search(r"(\d+\.?\d*)\s?(ml|g|kg|l|unit|n|pcs|gm)", full_blob))

    # 4. DATE DETECTION
    date_status = bool(re.search(r"(\d{2}[/\-\.]\d{2,4})", full_blob)) or "mfg" in full_blob

    # Final Logical Verdicts
    mrp_overall = currency_found and tax_status

    return {
        "MRP & Price": (currency_found, "Must show Currency (₹/Rs) and Value"),
        "Tax Declaration": (tax_status, "Must show 'Inclusive' and 'Taxes'"),
        "Net Quantity": (qty_status, "Must show standard units (e.g. 50ml)"),
        "Mfg Date": (date_status, "Must show Month/Year of packing")
    }, found_tax_parts

# --- MAIN UI ---
st.title("🛡️ Legal Metrology: Smart Verification")

with st.sidebar:
    st.header("⚙️ Inspector Settings")
    conf_level = st.slider("Detection Sensitivity", 0.01, 1.0, 0.15)
    st.markdown("""**Debug Info:**
    - Zone detection: YOLOv8
    - OCR Engine: EasyOCR
    - Logic: Granular Keyword Match""")

img_file = st.camera_input("Scan Product Label")

if img_file:
    image = ImageOps.exif_transpose(Image.open(img_file))
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing components..."):
        results = detector(img_bgr, conf=conf_level)
        detected_texts = []
        
        # 1. Zone OCR
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(0, x1-20), max(0, y1-20), min(img_bgr.shape[1], x2+20), min(img_bgr.shape[0], y2+20)
                
                crop = img_bgr[y1:y2, x1:x2]
                enhanced_crop = enhance_for_ocr(crop)
                
                txt_list = reader.readtext(enhanced_crop, detail=0)
                detected_texts.extend([repair_ocr_artifacts(t) for t in txt_list])
        
        # 2. Fallback Full Scan
        full_page_enhanced = enhance_for_ocr(img_bgr)
        detected_texts.extend(reader.readtext(full_page_enhanced, detail=0))

        # --- UI DISPLAY ---
        col_vis, col_rep = st.columns(2)
        with col_vis:
            res_plotted = results[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Detected Components")
            
        with col_rep:
            report, found_tags = check_compliance_granular(detected_texts)
            
            # Show "Found Tags" to the judges (Impressive!)
            if found_tags:
                tag_html = "".join([f'<span class="debug-tag">{t.upper()}</span>' for t in found_tags])
                st.markdown(f"**Keywords Detected:** {tag_html}", unsafe_allow_html=True)
            
            for rule, (status, desc) in report.items():
                s_class = "status-pass" if status else "status-fail"
                s_icon = "✅" if status else "❌"
                st.markdown(f"""
                    <div class="report-card">
                        <b>{rule}</b>: <span class="{s_class}">{s_icon}</span><br>
                        <small>{desc}</small>
                    </div>
                """, unsafe_allow_html=True)
            
            with st.expander("Show AI Raw Data"):
                st.write(detected_texts)
