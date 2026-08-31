import torch
# Fix for the PyTorch 2.6+ security serialization error
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

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Legal Metrology Compliance AI", layout="wide", page_icon="⚖️")

# Professional UI Styling for SIH Presentation
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .report-card { 
        background: white; padding: 20px; border-radius: 12px; 
        border-left: 10px solid #004085; color: #111111 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    .status-pass { color: #28a745 !important; font-weight: bold; font-size: 1.2em; }
    .status-fail { color: #dc3545 !important; font-weight: bold; font-size: 1.2em; }
    .card-title { color: #004085 !important; font-weight: bold; font-size: 1.3em; }
    .header-style { color: #004085; font-weight: bold; border-bottom: 2px solid #004085; padding-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- AI MODELS ---
@st.cache_resource
def load_ai():
    # Priority: Trained model (best.pt) > Default model (yolov8n.pt)
    model = YOLO('best.pt') if os.path.exists('best.pt') else YOLO('yolov8n.pt')
    # gpu=False ensures stability on CPU-only Streamlit Cloud instances
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

detector, reader = load_ai()

# --- OCR POST-PROCESSING ---
def repair_numbers(text):
    text = text.lower()
    # Post-processing map to fix common OCR character swaps on curved packaging
    mapping = {'s': '5', 'o': '0', 'i': '1', 'l': '1', 'b': '8', 'z': '2'}
    if any(char.isdigit() for char in text) or len(text) <= 4:
        for char, num in mapping.items():
            text = text.replace(char, num)
    return text

def enhance_image(img):
    # Convert to grayscale for OCR engine optimization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3x Zoom (Super-resolution) to handle micro-fonts on small containers
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    # Adaptive thresholding to eliminate shadows and plastic glare
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return enhanced

# --- COMPLIANCE ENGINE ---
def check_compliance(extracted_list):
    raw_blob = " ".join(extracted_list).lower()
    
    # 1. Price Verification
    price_marker = any(x in raw_blob for x in ['mrp', 'rs', 'price', 'retail', '₹', 'r5'])
    # Search for digits after repairing common swaps (e.g., S00 -> 500)
    digits_found = re.search(r"\d{2,}", repair_numbers(raw_blob))
    price_pass = price_marker and digits_found

    # 2. Tax Declaration (Fuzzy Matching Logic)
    # Required: 'Inclusive of all taxes' or variants like 'Incl. of all taxes'
    tax_score = fuzz.partial_ratio("inclusive of all taxes", raw_blob)
    tax_fragments = ["incl", "tax", "inc", "all", "taxes"]
    tax_pass = (tax_score > 45) or any(f in raw_blob for f in tax_fragments)

    # 3. Net Quantity Verification
    qty_pass = re.search(r"\d+\s?(ml|g|kg|l|n|unit|pcs|gm)", raw_blob)

    # 4. Manufacturing/Packing Date
    date_pass = re.search(r"(\d{2}[/\-\.]\d{2,4})", raw_blob) or "mfg" in raw_blob or "pkd" in raw_blob

    return {
        "MRP & Price Declaration": (bool(price_pass), "Rule 6: Mandatory MRP keyword and numeric price value"),
        "Tax Declaration": (bool(tax_pass), "Rule 6: Mandatory 'Inclusive of all taxes' phrase"),
        "Net Quantity": (bool(qty_pass), "Rule 7: Weight or Volume in standard metric units"),
        "Mfg/Packing Date": (bool(date_pass), "Rule 9: Month and Year of manufacture/packing")
    }

# --- MAIN INTERFACE ---
st.markdown("<h1 class='header-style'>⚖️ Legal Metrology Compliance AI</h1>", unsafe_allow_html=True)
st.write("Intelligent Verification System for Packaged Commodities Rules, 2011.")

# Camera input for mobile-friendly scanning
img_file = st.camera_input("Scan Product Label")

if img_file:
    # 1. Load and Fix Orientation
    image = ImageOps.exif_transpose(Image.open(img_file))
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("AI Analysis in Progress..."):
        # 2. YOLO Inference (Sensitivity fixed at 0.05 for maximum detection)
        results = detector(img_bgr, conf=0.05)
        detected_texts = []
        
        # 3. Targeted OCR (Zone Processing)
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Apply 40px Padding to ensure full strings are captured
                x1, y1, x2, y2 = max(0, x1-40), max(0, y1-40), min(img_bgr.shape[1], x2+40), min(img_bgr.shape[0], y2+40)
                
                crop = img_bgr[y1:y2, x1:x2]
                ocr_out = reader.readtext(enhance_image(crop), detail=0)
                detected_texts.extend(ocr_out)
        
        # 4. Fallback Full-Frame Scan (Heuristic Backup)
        detected_texts.extend(reader.readtext(enhance_image(img_bgr), detail=0))

        # --- RESULTS VISUALIZATION ---
        col_vis, col_rep = st.columns(2)
        
        with col_vis:
            st.subheader("Neural Detection View")
            # Convert BGR (OpenCV) to RGB (Streamlit)
            res_plotted = results[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col_rep:
            st.subheader("Compliance Report")
            report = check_compliance(detected_texts)
            
            for rule, (status, desc) in report.items():
                s_icon = "PASSED ✅" if status else "FAILED ❌"
                s_class = "status-pass" if status else "status-fail"
                
                st.markdown(f"""
                    <div class="report-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="card-title">{rule}</span>
                            <span class="{s_class}">{s_icon}</span>
                        </div>
                        <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eeeeee;">
                        <small><b>Requirement:</b> {desc}</small>
                    </div>
                """, unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.caption("Smart India Hackathon 2024 | Prototype for Legal Metrology Regulatory Compliance.")
