import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
from rapidfuzz import fuzz, process

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Metrology Inspector", layout="wide", page_icon="⚖️")

# --- PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 25px; color: #004085; }
    .report-card { 
        background: white; padding: 20px; border-radius: 12px; 
        border-left: 10px solid #004085; color: #111;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 15px;
    }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    .stCamera { border: 2px solid #004085; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- MODELS ---
@st.cache_resource
def load_models():
    # Detection: Using your trained model
    try:
        detector = YOLO('best.pt') 
    except:
        detector = YOLO('yolov8n.pt') 
    # Recognition: EasyOCR
    reader = easyocr.Reader(['en', 'hi']) # English + Hindi
    return detector, reader

detector, reader = load_models()

# --- LANGUAGE SETTINGS ---
lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi (हिन्दी)"])
T = {
    "English": {"title": "⚖️ Compliance Inspector", "scan": "Scan Label", "results": "Analysis Report", "mrp": "MRP & Taxes", "qty": "Net Quantity", "date": "Mfg Date"},
    "Hindi (हिन्दी)": {"title": "⚖️ अनुपालन निरीक्षक", "scan": "लेबल स्कैन करें", "results": "अनुपालन रिपोर्ट", "mrp": "MRP और कर", "qty": "शुद्ध मात्रा", "date": "निर्माण तिथि"}
}
txt = T[lang]

# --- FUZZY LOGIC COMPLIANCE ENGINE ---
def check_compliance(extracted_text):
    full_text = " ".join(extracted_text).lower()
    
    # Fuzzy match keywords to fix OCR typos (e.g., 'M1RP' instead of 'MRP')
    mrp_score = fuzz.partial_ratio("inclusive of all taxes", full_text)
    mrp_val = re.search(r"(?:mrp|rs|price)\.?\s?(\d+)", full_text)
    
    qty_val = re.search(r"(\d+\.?\d*)\s?(g|kg|ml|l|unit|n)", full_text)
    date_val = re.search(r"(\d{2}/\d{2,4})|(\d{2}-\d{2,4})", full_text)

    # Compliance Report
    return {
        txt["mrp"]: (mrp_score > 70 and mrp_val is not None, "Rule 6: Mandatory 'Inclusive of all taxes' phrase."),
        txt["qty"]: (qty_val is not None, "Rule 7: Must use standard metric units (g, kg, ml)."),
        txt["date"]: (date_val is not None, "Rule 9: Month and Year of packing must be visible.")
    }

# --- MAIN APP ---
st.title(txt["title"])
st.info("Automated checking for Legal Metrology (Packaged Commodities) Rules, 2011.")

img_file = st.camera_input(txt["scan"])

if img_file:
    # 1. Image Pre-processing
    image = Image.open(img_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("AI Pipeline Running..."):
        # 2. YOLO DETECTION (using your classes: batch vs label)
        results = detector(img_bgr, conf=0.15)
        detected_texts = []
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id] # 'stickers - batch' or 'stickers - label'
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_bgr[y1:y2, x1:x2]
                
                # Boost OCR for the "Batch" sticker specifically
                if "batch" in label:
                    crop = cv2.detailEnhance(crop, sigma_s=10, sigma_r=0.15)
                
                ocr_res = reader.readtext(crop, detail=0)
                detected_texts.extend(ocr_res)
        else:
            detected_texts = reader.readtext(img_bgr, detail=0)

        # 3. COMPLIANCE CHECK
        report = check_compliance(detected_texts)
        
        # 4. RESULTS DISPLAY
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("AI Vision")
            st.image(results[0].plot(), caption="Detected Zones", use_column_width=True)
            
        with col2:
            st.subheader(txt["results"])
            for rule, (passed, desc) in report.items():
                status = "PASS ✅" if passed else "FAIL ❌"
                color = "status-pass" if passed else "status-fail"
                st.markdown(f"""
                    <div class="report-card">
                        <div style="display:flex; justify-content:space-between;">
                            <b>{rule}</b>
                            <span class="{color}">{status}</span>
                        </div>
                        <div style="font-size:0.85em; color:#555; margin-top:5px;">{desc}</div>
                    </div>
                """, unsafe_allow_html=True)

        with st.expander("Show AI Raw Data"):
            st.write(detected_texts)
