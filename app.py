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
    # Only use YOLO if custom trained 'best.pt' exists
    try:
        detector = YOLO('best.pt') 
        has_custom_yolo = True
    except:
        detector = None
        has_custom_yolo = False
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    return detector, reader, has_custom_yolo

detector, reader, has_custom_yolo = load_ai_models()

# --- OCR PIPELINE FUNCTION ---
def run_smart_ocr(img_bgr):
    extracted_text = []
    annotated_img = img_bgr.copy()

    # 1. Full Horizontal Image OCR
    full_text_horiz = reader.readtext(img_bgr, detail=0)
    extracted_text.extend(full_text_horiz)

    # 2. Rotated Image OCR (To capture vertical text on borders like "NET CONTENT: 100 ml")
    img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    full_text_vert = reader.readtext(img_rotated, detail=0)
    extracted_text.extend(full_text_vert)

    # 3. Run Custom YOLO model if available
    if has_custom_yolo and detector is not None:
        results = detector(img_bgr, conf=0.3)
        if len(results[0].boxes) > 0:
            annotated_img = results[0].plot()
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    text = reader.readtext(gray_crop, detail=0)
                    extracted_text.extend(text)

    # Deduplicate extracted lines
    unique_text = list(dict.fromkeys(extracted_text))
    return unique_text, annotated_img

# --- COMPLIANCE ENGINE ---
def check_rules(text_list):
    full_text = " ".join(text_list).lower()
    
    # 1. Check MRP & Tax Inclusions
    has_mrp_kw = bool(re.search(r"(mrp|rs|retail|price|\u20b9)", full_text))
    has_tax_kw = bool(re.search(r"(incl|inclusive|tax|taxes)", full_text))
    mrp_status = has_mrp_kw and has_tax_kw

    # 2. Check Net Quantity (Flexible match for numbers + units like 100 ml, 100ml, g, kg)
    net_qty_status = bool(re.search(r"(\b\d+(\.\d+)?\s*(g|kg|ml|l|unit|pcs|n)\b)|(net\s*content)|(net\s*qty)", full_text))

    # 3. Check Mfg / Pkd / Expiry Date
    date_status = bool(re.search(r"(\d{2}/\d{2,4})|(mfd|pkd|mfg|batch|best before|exp)", full_text))

    report = {
        "MRP & Taxes": (mrp_status, "Rule 6: MRP declaration must include 'Inclusive of all taxes'"),
        "Net Quantity": (net_qty_status, "Rule 7: Standard declaration of Net Quantity (g, kg, ml, l)"),
        "Mfg/Pkd Date": (date_status, "Rule 9: Month & Year of packing/manufacture must be declared")
    }
    return report

# --- UI LAYOUT ---
st.title("🛡️ Automated Metrology Compliance")
st.write("Target: Automated Compliance Verification for Packaged Goods")

img_input = st.camera_input("Scan Product Label")

if img_input:
    image = Image.open(img_input)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing image, scanning angles, and verifying compliance..."):
        extracted_text, annotated_img = run_smart_ocr(img_bgr)
        compliance_results = check_rules(extracted_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Processing View")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Scanned Region")
            
        with col2:
            st.subheader("2. Compliance Report")
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

    with st.expander("Show Raw Extracted OCR Text"):
        st.write(extracted_text)
