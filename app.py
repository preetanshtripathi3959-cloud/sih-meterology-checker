import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import re

# --- UI SETTINGS ---
st.set_page_config(page_title="SIH: Smart OCR System", layout="wide")

st.markdown("""
    <style>
    .report-card { background: #ffffff; padding: 15px; border-radius: 10px; border-left: 10px solid #004085; color: black; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- MODELS ---
@st.cache_resource
def load_ai_models():
    # Detection Stage: YOLOv8
    try:
        detector = YOLO('best.pt') # Your trained model
    except:
        detector = YOLO('yolov8n.pt') # Fallback
    
    # Recognition Stage: OCR
    reader = easyocr.Reader(['en'])
    return detector, reader

detector, reader = load_ai_models()

# --- THE OCR PIPELINE FUNCTION ---
def run_smart_ocr(img_bgr):
    # Step 1: Detect Text Blocks using YOLO
    results = detector(img_bgr, conf=0.3)
    
    all_extracted_text = []
    
    # Step 2: Loop through detected blocks and run OCR on each
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Crop the detection for OCR
            crop = img_bgr[y1:y2, x1:x2]
            # Convert to gray for better OCR recognition
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Recognize text in the crop
            text = reader.readtext(gray_crop, detail=0)
            all_extracted_text.extend(text)
    else:
        # Fallback: OCR the whole image if no blocks detected
        all_extracted_text = reader.readtext(img_bgr, detail=0)
        
    return all_extracted_text, results[0].plot()

# --- COMPLIANCE ENGINE ---
def check_rules(text_list):
    full_text = " ".join(text_list).lower()
    
    # Matching rules from Legal Metrology Act
    report = {
        "MRP & Taxes": (re.search(r"(mrp|rs|retail|price).?\d+", full_text) and "incl" in full_text, 
                        "Rule 6: MRP must include 'Inclusive of all taxes'"),
        "Net Quantity": (re.search(r"(\d+)\s?(g|kg|ml|l|unit|n)", full_text), 
                        "Rule 7: Standard units (g, kg, ml, l) are mandatory"),
        "Mfg/Pkd Date": (re.search(r"\d{2}/\d{2,4}", full_text) or "pkd" in full_text, 
                        "Rule 9: Month and Year of packing must be declared")
    }
    return report

# --- UI LAYOUT ---
st.title("🛡️ Automated Metrology Compliance")
st.write("Target: Problem Statement - Automated Compliance for Packaged Commodities")

img_input = st.camera_input("Scan Product Label")

if img_input:
    # Prepare Image
    image = Image.open(img_input)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("AI Pipeline: Detecting ➔ Recognizing ➔ Verifying..."):
        # Run the Two-Stage OCR
        extracted_text, annotated_img = run_smart_ocr(img_bgr)
        
        # Verify Rules
        compliance_results = check_rules(extracted_text)
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. AI Detection")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="YOLOv8 Detected Zones")
            
        with col2:
            st.subheader("2. OCR & Compliance Report")
            for rule, (status, desc) in compliance_results.items():
                st_icon = "✅" if status else "❌"
                st_color = "status-pass" if status else "status-fail"
                st.markdown(f"""
                    <div class="report-card">
                        <b>{rule}</b><br>
                        <span class="{st_color}">{st_icon} {"PASSED" if status else "VIOLATION"}</span><br>
                        <small>{desc}</small>
                    </div>
                """, unsafe_allow_html=True)

    with st.expander("Show Extracted OCR Text"):
        st.write(extracted_text)
