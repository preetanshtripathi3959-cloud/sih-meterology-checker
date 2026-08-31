import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
import os
from rapidfuzz import fuzz

# --- UI CONFIG ---
st.set_page_config(page_title="SIH Compliance AI", layout="wide")

st.markdown("""
    <style>
    .report-card { background: white; padding: 15px; border-radius: 10px; border-left: 10px solid #004085; color: black; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .status-pass { color: green; font-weight: bold; }
    .status-fail { color: red; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    if os.path.exists('best.pt'):
        model = YOLO('best.pt')
        status = "✅ Custom Model Loaded"
    else:
        model = YOLO('yolov8n.pt')
        status = "⚠️ Generic Model Loaded"
    reader = easyocr.Reader(['en'])
    return model, reader, status

detector, reader, model_status = load_models()

# --- SIDEBAR ---
st.sidebar.title("🛠️ AI Debugger")
conf_val = st.sidebar.slider("Confidence", 0.0, 1.0, 0.35) # Raised to 0.35 to avoid detecting faces/shirts
st.sidebar.write(f"Model: {model_status}")

# --- UPDATED COMPLIANCE LOGIC (Better Regex) ---
def check_compliance(text_list):
    full_text = " ".join(text_list).lower()
    
    # 1. MRP & Taxes: Look for MRP keyword and 'inclusive'
    mrp_found = re.search(r"(mrp|rs|retail|price).?\s?\d+", full_text)
    tax_phrase = fuzz.partial_ratio("inclusive of all taxes", full_text) > 60
    
    # 2. Net Quantity: Handles "50ml", "1N", "1 N x 50ml"
    qty_found = re.search(r"(\d+)\s?(g|kg|ml|l|unit|n|pcs)", full_text)
    
    # 3. Date: Handles 01-05-2026, 01/05/2026, 01.05.2026
    date_found = re.search(r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})", full_text) or "mfg" in full_text or "pkd" in full_text

    return {
        "MRP & Taxes": (mrp_found and tax_phrase, "Required: MRP + 'inclusive of all taxes'"),
        "Net Quantity": (bool(qty_found), "Required: Weight/Volume (e.g. 50ml)"),
        "Mfg/Pkd Date": (bool(date_found), "Required: Month/Year (e.g. 05-2026)")
    }

# --- MAIN APP ---
st.title("🛡️ Legal Metrology Inspector")
img_file = st.camera_input("Scan Product Label")

if img_file:
    img = Image.open(img_file)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    with st.spinner("AI Processing..."):
        results = detector(img_bgr, conf=conf_val)
        detected_texts = []
        
        # If YOLO finds boxes, OCR them
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_bgr[y1:y2, x1:x2]
                # Light enhancement for OCR
                crop = cv2.convertScaleAbs(crop, alpha=1.5, beta=0) 
                txt = reader.readtext(crop, detail=0)
                detected_texts.extend(txt)
        
        # FALLBACK: Always OCR the whole image too to ensure we don't miss anything!
        full_page_text = reader.readtext(img_bgr, detail=0)
        detected_texts.extend(full_page_text)

        # Remove duplicates while keeping order
        detected_texts = list(dict.fromkeys(detected_texts))

        # Show Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("AI Vision")
            res_plotted = results[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
            
        with col2:
            st.subheader("Compliance Report")
            report = check_compliance(detected_texts)
            for rule, (status, desc) in report.items():
                st_color = "status-pass" if status else "status-fail"
                st_icon = "✅" if status else "❌"
                st.markdown(f'<div class="report-card"><b>{rule}</b>: <span class="{st_color}">{st_icon} {"PASS" if status else "FAIL"}</span><br><small>{desc}</small></div>', unsafe_allow_html=True)
            
            with st.expander("Show Scanned Text"):
                st.write(detected_texts)
