import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
from rapidfuzz import fuzz

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Metrology Inspector", layout="wide", page_icon="⚖️")

# --- CUSTOM UI STYLING ---
st.markdown("""
    <style>
    .report-card { 
        background: #ffffff; padding: 20px; border-radius: 12px; 
        border-left: 10px solid #004085; color: #111111 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px;
    }
    .status-pass { color: #28a745 !important; font-weight: bold; }
    .status-fail { color: #dc3545 !important; font-weight: bold; }
    .card-title { color: #004085 !important; font-weight: bold; font-size: 1.2em; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD AI MODELS ---
@st.cache_resource
def load_models():
    # Detection: YOLOv8 (Trained on stickers - batch, stickers - label)
    try:
        detector = YOLO('best.pt') 
    except:
        detector = YOLO('yolov8n.pt') # Fallback if best.pt is missing
    
    # Recognition: EasyOCR (English + Hindi)
    reader = easyocr.Reader(['en', 'hi'])
    return detector, reader

detector, reader = load_models()

# --- LANGUAGE DICTIONARY ---
with st.sidebar:
    st.title("⚙️ Settings")
    lang_choice = st.selectbox("Language / भाषा", ["English", "Hindi (हिन्दी)"])

T = {
    "English": {
        "title": "⚖️ Compliance Dashboard",
        "info": "Checking Legal Metrology Rules, 2011",
        "scan": "Scan Product Label",
        "mrp": "MRP & Tax Declaration",
        "qty": "Net Quantity Check",
        "date": "Mfg/Packing Date",
        "pass": "PASSED ✅",
        "fail": "VIOLATION ❌",
        "req": "Requirement"
    },
    "Hindi (हिन्दी)": {
        "title": "⚖️ अनुपालन डैशबोर्ड",
        "info": "कानूनी मेट्रोलॉजी नियम, 2011 की जांच",
        "scan": "लेबल स्कैन करें",
        "mrp": "MRP और कर घोषणा",
        "qty": "शुद्ध मात्रा की जांच",
        "date": "निर्माण की तारीख",
        "pass": "पास ✅",
        "fail": "उल्लंघन ❌",
        "req": "आवश्यकता"
    }
}
L = T[lang_choice]

# --- COMPLIANCE ENGINE ---
def check_rules(extracted_text):
    full_text = " ".join(extracted_text).lower()
    
    # Fuzzy Match for the mandatory phrase
    mrp_phrase_score = fuzz.partial_ratio("inclusive of all taxes", full_text)
    mrp_found = re.search(r"(?:mrp|rs|price)\.?\s?(\d+)", full_text)
    
    qty_found = re.search(r"(\d+\.?\d*)\s?(g|kg|ml|l|unit|n)", full_text)
    date_found = re.search(r"(\d{2}/\d{2,4})|(\d{2}-\d{2,4})", full_text)

    return {
        "MRP": (mrp_phrase_score > 75 and mrp_found, "Rule 6: Must include 'Inclusive of all taxes'"),
        "QTY": (bool(qty_found), "Rule 7: Standard metric units (g, kg, ml, l) required"),
        "DATE": (bool(date_found) or "pkd" in full_text or "mfd" in full_text, "Rule 9: Month & Year of packing required")
    }

# --- MAIN INTERFACE ---
st.title(L["title"])
st.caption(L["info"])

img_file = st.camera_input(L["scan"])

if img_file:
    # Convert image for YOLO
    image = Image.open(img_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("AI Pipeline Processing..."):
        # 1. YOLO DETECTION (conf=0.15 for better sensitivity)
        results = detector(img_bgr, conf=0.15)
        detected_texts = []
        
        # 2. CROP & OCR
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label_name = results[0].names[cls_id]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_bgr[y1:y2, x1:x2]
                
                # Special enhancement for 'stickers - batch' (contains small text)
                if "batch" in label_name:
                    crop = cv2.detailEnhance(crop, sigma_s=10, sigma_r=0.15)
                
                ocr_out = reader.readtext(crop, detail=0)
                detected_texts.extend(ocr_out)
        else:
            # Fallback to full page OCR if YOLO misses
            detected_texts = reader.readtext(img_bgr, detail=0)

        # 3. RULE ANALYSIS
        final_results = check_rules(detected_texts)
        
        # 4. UI DISPLAY
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("AI Vision")
            # --- FIX: Convert BGR to RGB for Streamlit ---
            annotated_frame = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected Zones", use_container_width=True)
            
        with col2:
            st.subheader("Compliance Report")
            
            # Map keys to translated labels
            ui_labels = {"MRP": L["mrp"], "QTY": L["qty"], "DATE": L["date"]}
            
            for key, (passed, desc) in final_results.items():
                status_txt = L["pass"] if passed else L["fail"]
                status_class = "status-pass" if passed else "status-fail"
                
                st.markdown(f"""
                    <div class="report-card">
                        <div style="display:flex; justify-content:space-between;">
                            <span class="card-title">{ui_labels[key]}</span>
                            <span class="{status_class}">{status_txt}</span>
                        </div>
                        <div style="margin-top:8px; font-size:0.9em;">
                            <b>{L['req']}:</b> {desc}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Debugging Expandable
    with st.expander("Show AI Raw Metadata"):
        st.write(detected_texts)
