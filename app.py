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

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .report-card { background: white; padding: 15px; border-radius: 10px; border-left: 10px solid #004085; color: black; margin-bottom: 10px; }
    .status-pass { color: #28a745 !important; font-weight: bold; }
    .status-fail { color: #dc3545 !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_ai():
    model = YOLO('best.pt') if os.path.exists('best.pt') else YOLO('yolov8n.pt')
    # gpu=False to save memory on Streamlit Cloud
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, reader = load_ai()

# --- OCR REPAIR SYSTEM (Fixes common misreads) ---
def repair_text(text):
    # Fix common OCR errors for Legal Metrology
    text = text.replace('M1RP', 'MRP').replace('Mzp', 'MRP').replace('MRP:', 'MRP')
    text = text.replace('1nclusive', 'inclusive').replace('a11', 'all').replace('1ncl', 'incl')
    text = text.replace('O', '0') # Often reads zero as capital O
    return text

# --- IMAGE ENHANCEMENT (High Performance) ---
def advanced_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Upscale 3x (Crucial for tiny bottle text)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    # Adaptive Thresholding to handle shadows
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return enhanced

# --- RULES ENGINE ---
def check_compliance(extracted_list):
    full_blob = " ".join(extracted_list).lower()
    full_blob = repair_text(full_blob)
    
    # 1. MRP Check (Fuzzy + Regex)
    # Looks for 'mrp', 'rs', 'price' or the symbol ₹
    mrp_keywords = ["mrp", "rs", "price", "retail", "maximum"]
    found_mrp_word = any(kw in full_blob for kw in mrp_keywords) or "₹" in full_blob
    
    # Check for the mandatory "Inclusive of all taxes" phrase
    tax_score = fuzz.partial_ratio("inclusive of all taxes", full_blob)
    tax_short_score = fuzz.partial_ratio("incl of all taxes", full_blob)
    
    mrp_status = (found_mrp_word or re.search(r"\d+\.\d{2}", full_blob)) and (max(tax_score, tax_short_score) > 60)

    # 2. Qty Check
    qty_status = bool(re.search(r"(\d+)\s?(ml|g|kg|l|unit|n|pcs|gm)", full_blob))

    # 3. Date Check
    date_status = bool(re.search(r"(\d{2}[/\-\.]\d{2,4})", full_blob)) or "mfg" in full_blob or "pkd" in full_blob

    return {
        "MRP & Taxes": (mrp_status, "Rule 6: MRP and 'Inclusive of all taxes'"),
        "Net Quantity": (qty_status, "Rule 7: Net Weight/Volume (e.g. 50ml)"),
        "Mfg Date": (date_status, "Rule 9: Month & Year of packing")
    }

# --- MAIN UI ---
st.title("⚖️ Legal Metrology Inspector")

with st.sidebar:
    st.header("⚙️ Debugger")
    conf_level = st.slider("Detection Sensitivity", 0.01, 1.0, 0.15)
    st.info("Scan the 'Back of the Pack' for results.")

img_file = st.camera_input("Scan Label")

if img_file:
    # 1. Prepare Image
    image = ImageOps.exif_transpose(Image.open(img_file))
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("AI analyzing..."):
        # 2. YOLO Detection
        results = detector(img_bgr, conf=conf_level)
        detected_texts = []
        
        # 3. OCR Stage with 3x Zoom
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add Padding to prevent cutting text
                x1, y1, x2, y2 = max(0, x1-20), max(0, y1-20), min(img_bgr.shape[1], x2+20), min(img_bgr.shape[0], y2+20)
                
                crop = img_bgr[y1:y2, x1:x2]
                enhanced_crop = advanced_enhance(crop)
                
                # Run OCR
                txt_list = reader.readtext(enhanced_crop, detail=0)
                repaired = [repair_text(t) for t in txt_list]
                detected_texts.extend(repaired)
        
        # 4. Fallback Full Page Scan (Aggressive)
        full_page_enhanced = advanced_enhance(img_bgr)
        detected_texts.extend(reader.readtext(full_page_enhanced, detail=0))

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True, caption="Zones Detected")
            
        with col2:
            report = check_compliance(detected_texts)
            for rule, (status, desc) in report.items():
                s_class = "status-pass" if status else "status-fail"
                s_icon = "✅" if status else "❌"
                st.markdown(f"""
                    <div class="report-card">
                        <b>{rule}</b>: <span class="{s_class}">{s_icon}</span><br>
                        <small>{desc}</small>
                    </div>
                """, unsafe_allow_html=True)
            
            with st.expander("Show Scanned Data (Debugger)"):
                st.write(detected_texts)
