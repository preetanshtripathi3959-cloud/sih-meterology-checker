import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
from rapidfuzz import fuzz

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Metrology Inspector", layout="wide")

# --- UI STYLING ---
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
    # Attempt to load your custom model
    try:
        # Check if best.pt exists in the root directory
        import os
        if os.path.exists('best.pt'):
            detector = YOLO('best.pt')
            model_type = "Custom (best.pt)"
        else:
            detector = YOLO('yolov8n.pt')
            model_type = "Generic (yolov8n.pt)"
    except Exception as e:
        detector = YOLO('yolov8n.pt')
        model_type = f"Error loading best.pt: {e}"
        
    reader = easyocr.Reader(['en'])
    return detector, reader, model_type

detector, reader, model_info = load_models()

# Sidebar Debug Info
st.sidebar.title("🛠️ AI Diagnostics")
st.sidebar.write(f"**Model Loaded:** {model_info}")

# --- COMPLIANCE LOGIC ---
def check_compliance(text_list):
    full_text = " ".join(text_list).lower()
    
    # Using Fuzzy matching for better results
    mrp_found = re.search(r"(mrp|rs|retail|price).?\d+", full_text)
    tax_phrase = fuzz.partial_ratio("inclusive of all taxes", full_text) > 60
    
    qty_found = re.search(r"(\d+)\s?(g|kg|ml|l|unit|n)", full_text)
    date_found = re.search(r"\d{2}/\d{2,4}", full_text) or "pkd" in full_text or "mfd" in full_text

    return {
        "MRP & Taxes": (mrp_found and tax_phrase, "Mandatory: MRP + 'Inclusive of all taxes'"),
        "Net Quantity": (bool(qty_found), "Mandatory: Standard units (g, kg, ml, l)"),
        "Mfg Date": (bool(date_found), "Mandatory: Month and Year of packing")
    }

# --- MAIN APP ---
st.title("⚖️ Legal Metrology Compliance AI")
img_file = st.camera_input("Scan Product Label")

if img_file:
    # 1. Prepare Image
    image = Image.open(img_file)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing..."):
        # 2. YOLO DETECTION (Force sensitivity with conf=0.05)
        results = detector(img_bgr, conf=0.05)
        detected_texts = []
        
        # 3. OCR Stage
        if len(results[0].boxes) > 0:
            st.sidebar.success(f"Detected {len(results[0].boxes)} zones!")
            for box in results[0].boxes:
                # Get class name
                cls = int(box.cls[0])
                name = results[0].names[cls]
                conf = float(box.conf[0])
                st.sidebar.write(f"Zone: {name} ({conf:.2f}%)")
                
                # Crop and OCR
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_bgr[y1:y2, x1:x2]
                
                # Sharpness boost
                crop = cv2.convertScaleAbs(crop, alpha=1.2, beta=10)
                
                txt = reader.readtext(crop, detail=0)
                detected_texts.extend(txt)
        else:
            st.sidebar.warning("No zones detected. Scanning entire image...")
            detected_texts = reader.readtext(img_np, detail=0)

        # 4. RESULTS
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("AI Vision")
            # FIX: Convert to RGB for Streamlit display
            annotated_frame = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
            
        with col2:
            st.subheader("Compliance Report")
            report = check_compliance(detected_texts)
            for rule, (status, desc) in report.items():
                st_color = "status-pass" if status else "status-fail"
                st_icon = "✅" if status else "❌"
                st.markdown(f"""
                    <div class="report-card">
                        <b>{rule}</b><br>
                        <span class="{st_color}">{st_icon} {"PASSED" if status else "VIOLATION"}</span><br>
                        <small>{desc}</small>
                    </div>
                """, unsafe_allow_html=True)
                
        with st.expander("Detected Text (Raw)"):
            st.write(detected_texts)
