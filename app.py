import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="SIH AI Inspector", layout="wide")

# --- UI STYLING ---
st.markdown("""
    <style>
    .report-card { background: white; padding: 15px; border-radius: 10px; border-left: 10px solid #004085; color: black; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .status-pass { color: green; font-weight: bold; }
    .status-fail { color: red; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR DIAGNOSTICS ---
st.sidebar.title("🛠️ AI Debugger")
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.15)
st.sidebar.info("Lower this slider if no boxes appear.")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # 1. Check if best.pt exists
    if os.path.exists('best.pt'):
        try:
            model = YOLO('best.pt')
            status = "✅ Custom Model (best.pt) Loaded"
            classes = model.names
        except Exception as e:
            model = YOLO('yolov8n.pt')
            status = f"❌ Error loading best.pt: {e}"
            classes = "N/A"
    else:
        model = YOLO('yolov8n.pt')
        status = "⚠️ best.pt NOT FOUND. Using default model."
        classes = "N/A"
        
    reader = easyocr.Reader(['en'])
    return model, reader, status, classes

detector, reader, model_status, model_classes = load_models()

st.sidebar.write(f"**Status:** {model_status}")
st.sidebar.write(f"**Detected Classes:** {model_classes}")

# --- MAIN UI ---
st.title("🛡️ Legal Metrology Compliance AI")
st.write("Scan labels to verify MRP, Dates, and Quantity.")

img_file = st.camera_input("Take a photo of the product label")

if img_file:
    image = Image.open(img_file)
    img_np = np.array(image)
    # Convert RGB (Streamlit) to BGR (OpenCV/YOLO)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Inference in progress..."):
        # RUN DETECTION with the slider value
        results = detector(img_bgr, conf=conf_threshold)
        
        detected_texts = []
        
        if len(results[0].boxes) > 0:
            st.sidebar.success(f"Found {len(results[0].boxes)} objects!")
            for box in results[0].boxes:
                # Crop and OCR
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_bgr[y1:y2, x1:x2]
                
                # Pre-process crop for OCR accuracy
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                txt = reader.readtext(enhanced, detail=0)
                detected_texts.extend(txt)
        else:
            st.sidebar.error("No objects detected. Try lowering the Confidence Threshold.")
            # Fallback to full page OCR
            detected_texts = reader.readtext(img_np, detail=0)

    # --- DISPLAY ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AI Vision")
        # YOLO plot is BGR, convert to RGB for Streamlit
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_rgb, use_container_width=True)
        
    with col2:
        st.subheader("Compliance Report")
        full_text = " ".join(detected_texts).lower()
        
        # Rules Logic
        mrp_ok = re.search(r"(mrp|rs|retail|price).?\d+", full_text) and ("incl" in full_text or "tax" in full_text)
        qty_ok = re.search(r"(\d+)\s?(g|kg|ml|l|unit|n)", full_text)
        date_ok = re.search(r"\d{2}/\d{2,4}", full_text) or "pkd" in full_text or "mfd" in full_text

        checks = [("MRP & Taxes", mrp_ok), ("Net Quantity", qty_ok), ("Mfg/Pkd Date", date_ok)]
        
        for name, passed in checks:
            color = "status-pass" if passed else "status-fail"
            icon = "✅" if passed else "❌"
            st.markdown(f'<div class="report-card"><b>{name}</b>: <span class="{color}">{icon} {"PASS" if passed else "FAIL"}</span></div>', unsafe_allow_html=True)
            
        with st.expander("Show Scanned Data"):
            st.write(detected_texts)
