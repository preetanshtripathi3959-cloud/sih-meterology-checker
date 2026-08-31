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
    .card-title { color: #004085 !important; font-weight: bold; font-size: 1.1em; }
    .debug-tag { background: #e9ecef; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.8em; margin-right: 5px; color: #333; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ai():
    model = YOLO('best.pt') if os.path.exists('best.pt') else YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, reader = load_ai()

def repair_ocr_for_numbers(text):
    # OCR often swaps numbers and letters on small bottles
    # 5 -> S, 0 -> O, 1 -> I/l, 8 -> B
    text = text.lower()
    mapping = {'s': '5', 'o': '0', 'i': '1', 'l': '1', 'b': '8', 'z': '2'}
    # Only swap if the word looks like it should be a number
    if any(char.isdigit() for char in text) or len(text) <= 4:
        for char, num in mapping.items():
            text = text.replace(char, num)
    return text

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3x Zoom
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    # Adaptive thresholding
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return enhanced

def check_compliance_ultra_flexible(extracted_list):
    # Join text and repair common errors
    raw_blob = " ".join(extracted_list).lower()
    
    # 1. PRICE DETECTION (More aggressive)
    # Look for any currency marker OR the word 'mrp'
    price_marker = any(x in raw_blob for x in ['mrp', 'rs', 'price', 'retail', '₹', 'r5'])
    # Look for any number that looks like a price (e.g., 500, 500.00, 500-)
    digits_found = re.search(r"\d{2,}", repair_ocr_for_numbers(raw_blob))
    price_pass = price_marker and digits_found

    # 2. TAX DETECTION (If it sees 'incl' or 'tax', it passes)
    tax_score = fuzz.partial_ratio("inclusive of all taxes", raw_blob)
    tax_fragments = ["incl", "tax", "taxe", "inc", "all"]
    found_frags = [f for f in tax_fragments if f in raw_blob]
    tax_pass = (tax_score > 45) or (len(found_frags) >= 1)

    # 3. QTY & DATE
    # Look for numbers near ml, g, kg
    qty_pass = re.search(r"\d+\s?(ml|g|kg|l|n|unit|pcs|gm)", raw_blob)
    date_pass = re.search(r"(\d{2}[/\-\.]\d{2,4})", raw_blob) or "mfg" in raw_blob or "pkd" in raw_blob

    return {
        "MRP & Price": (bool(price_pass), "MRP/Rs keyword AND price digits"),
        "Tax Declaration": (bool(tax_pass), "Mandatory 'Inclusive of all taxes' phrase"),
        "Net Quantity": (bool(qty_pass), "Weight/Volume (e.g. 50ml)"),
        "Mfg/Pkd Date": (bool(date_pass), "Month/Year of packing")
    }, found_frags, digits_found.group(0) if digits_found else "None"

# --- UI ---
st.title("🛡️ Legal Metrology: Smart Inspector")

with st.sidebar:
    st.header("⚙️ Settings")
    conf_level = st.slider("Sensitivity", 0.01, 1.0, 0.15)

img_file = st.camera_input("Scan Label")

if img_file:
    image = ImageOps.exif_transpose(Image.open(img_file))
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("AI Analysis..."):
        results = detector(img_bgr, conf=conf_level)
        detected_texts = []
        
        # Zone OCR
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Huge padding to grab surrounding context
                x1, y1, x2, y2 = max(0, x1-40), max(0, y1-40), min(img_bgr.shape[1], x2+40), min(img_bgr.shape[0], y2+40)
                crop = img_bgr[y1:y2, x1:x2]
                txt = reader.readtext(enhance_image(crop), detail=0)
                detected_texts.extend(txt)
        
        # Fallback Full Scan
        detected_texts.extend(reader.readtext(enhance_image(img_bgr), detail=0))

        col_v, col_r = st.columns(2)
        with col_v:
            st.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col_r:
            report, fragments, price_val = check_compliance_ultra_flexible(detected_texts)
            
            # RESULTS DASHBOARD
            st.write(f"**Detected Price Value:** `{price_val}`")
            if fragments:
                tags = "".join([f'<span class="debug-tag">{f.upper()}</span>' for f in fragments])
                st.markdown(f"**Detected Tax Tokens:** {tags}", unsafe_allow_html=True)

            st.divider()

            for rul
