import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageOps
import re
import os
from rapidfuzz import fuzz, process

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SIH: Legal Metrology AI", layout="wide", page_icon="⚖️")

# --- PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
    .report-card { 
        background: white; padding: 18px; border-radius: 12px; 
        border-left: 10px solid #004085; color: #111111 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px;
    }
    .status-pass { color: #28a745 !important; font-weight: bold; }
    .status-fail { color: #dc3545 !important; font-weight: bold; }
    .card-title { color: #004085 !important; font-weight: bold; font-size: 1.1em; }
    .debug-text { font-family: monospace; font-size: 0.8em; }
    </style>
""", unsafe_allow_html=True)

# --- MODELS ---
@st.cache_resource
def load_ai_tools():
    # Load YOLO
    if os.path.exists('best.pt'):
        model = YOLO('best.pt')
        m_status = "✅ Custom Model (best.pt) Active"
    else:
        model = YOLO('yolov8n.pt')
        m_status = "⚠️ Generic YOLOv8 Active"
    
    # Load OCR
    reader = easyocr.Reader(['en'])
    return model, reader, m_status

detector, reader, model_status = load_ai_tools()

# --- HEAVY DUTY IMAGE ENHANCEMENT ---
def enhance_for_ocr(img_crop):
    # 1. Convert to Gray
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    # 2. Rescale (Zoom in 2x) - This makes small text readable
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # 3. Sharpening Filter
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
    return sharpened

# --- FUZZY LOGIC ENGINE ---
def run_compliance_check(text_list):
    full_blob = " ".join(text_list).lower()
    
    # 1. MRP & Taxes: Fuzzy check for "Inclusive of all taxes"
    tax_similarity = fuzz.partial_ratio("inclusive of all taxes", full_blob)
    mrp_regex = re.search(r"(mrp|rs|retail|price|max).?\s?\d+", full_blob)
    mrp_status = (tax_similarity > 65) or bool(mrp_regex)

    # 2. Net Quantity: Handles "50ml", "1N", "100g"
    qty_regex = re.search(r"(\d+\.?\d*)\s?(ml|g|kg|l|unit|n|pcs|gm)", full_blob)
    
    # 3. Date: Handles 01-05-2026, 05/2026, Mfg Date
    date_regex = re.search(r"(\d{2}[/\-\.]\d{2,4})", full_blob)
    date_status = bool(date_regex) or "mfg" in full_blob or "pkd" in full_blob

    return {
        "MRP & Taxes": (mrp_status, "Rule 6: MRP and 'Inclusive of all taxes' phrase"),
        "Net Quantity": (bool(qty_regex), "Rule 7: Weight/Volume (e.g., 50ml, 100g)"),
        "Mfg/Pkd Date": (date_status, "Rule 9: Month and Year of packing")
    }

# --- SIDEBAR DIAGNOSTICS ---
with st.sidebar:
    st.title("🛠️ AI Debugger")
    st.write(f"**Status:** {model_status}")
    conf_threshold = st.sidebar.slider("Confidence Slider", 0.01, 1.0, 0.15)
    st.info("💡 Tip: For small perfume bottles, keep the phone steady and 6 inches away.")
    st.divider()
    st.write("Detected Classes:")
    st.json(detector.names)

# --- MAIN APP UI ---
st.title("🛡️ Legal Metrology Compliance AI")
st.write("Scan packaged commodities to verify Rule 2011 compliance.")

img_file = st.camera_input("Scan Product Label")

if img_file:
    # Prepare Image
    raw_img = Image.open(img_file)
    raw_img = ImageOps.exif_transpose(raw_img) # Fix auto-rotation
    img_bgr = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Stage 1: Detecting Zones..."):
        results = detector(img_bgr, conf=conf_threshold)
        detected_texts = []
        
        # Process YOLO Detections
        if len(results[0].boxes) > 0:
            st.sidebar.success(f"AI found {len(results[0].boxes)} zones!")
            for box in results[0].boxes:
                # Get coords
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Add Padding (15px) so we don't cut off text edges
                h, w = img_bgr.shape[:2]
                x1, y1 = max(0, x1-15), max(0, y1-15)
                x2, y2 = min(w, x2+15), min(h, y2+15)
                
                # Crop and Enhance
                crop = img_bgr[y1:y2, x1:x2]
                enhanced_crop = enhance_for_ocr(crop)
                
                # OCR
                txt = reader.readtext(enhanced_crop, detail=0)
                detected_texts.extend(txt)
        
        # Stage 2: Fallback (Scan full image with enhancement)
        full_enhanced = enhance_for_ocr(img_bgr)
        full_txt = reader.readtext(full_enhanced, detail=0)
        detected_texts.extend(full_txt)

        # Remove duplicates
        detected_texts = list(dict.fromkeys(detected_texts))

        # --- DISPLAY RESULTS ---
        col_vis, col_rep = st.columns(2)
        
        with col_vis:
            st.subheader("AI Vision")
            # Convert BGR to RGB for Streamlit
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Detected Regulatory Zones", use_container_width=True)
            
        with col_rep:
            st.subheader("Compliance Report")
            report = run_compliance_check(detected_texts)
            
            for rule, (status, desc) in report.items():
                s_icon = "✅" if status else "❌"
                s_text = "PASS" if status else "FAIL"
                s_class = "status-pass" if status else "status-fail"
                
                st.markdown(f"""
                    <div class="report-card">
                        <div style="display:flex; justify-content:space-between;">
                            <span class="card-title">{rule}</span>
                            <span class="{s_class}">{s_text} {s_icon}</span>
                        </div>
                        <div style="font-size:0.85em; margin-top:5px; color:#555;">
                            <b>Requirement:</b> {desc}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with st.expander("Show AI Raw Data (Debug Mode)"):
                st.write(detected_texts)
