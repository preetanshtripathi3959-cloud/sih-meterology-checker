import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
from rapidfuzz import fuzz

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Legal Metrology AI", layout="wide")

# --- FIXED CSS (Visibility for Dark & Light Mode) ---
st.markdown("""
    <style>
    .report-card {
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px;
        border-left: 10px solid #004085;
        margin-bottom: 20px;
        color: #111111 !important; /* Forces black text */
    }
    .card-title { color: #004085 !important; font-weight: bold; font-size: 1.4em; }
    .status-pass { color: #1e7e34 !important; font-weight: bold; font-size: 1.2em; }
    .status-fail { color: #bd2130 !important; font-weight: bold; font-size: 1.2em; }
    .recommendation { color: #444444 !important; font-size: 0.9em; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- ACCURACY BOOST: ADVANCED IMAGE PRE-PROCESSING ---
def enhance_for_ocr(img_array):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 2. Rescale image (OCR works better if text is a specific size)
    height, width = gray.shape
    scale_factor = 1.5 if width < 1000 else 1.0
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 3. Apply Bilateral Filter (Removes noise but keeps edges sharp)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. Adaptive Thresholding (Handles uneven lighting/shadows)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return thresh

# --- LOGIC ENGINE (More flexible Regex) ---
def run_compliance_logic(text_list):
    full_blob = " ".join(text_list).lower()
    
    # Improved Regex & Fuzzy Logic
    mrp_pattern = r"(mrp|rs|retail|price|max).?\s?(\d+)"
    qty_pattern = r"(\d+\.?\d*)\s?(g|kg|ml|l|unit|n|pcs|gm|mtr)"
    date_pattern = r"(\d{2}/\d{2,4})|(\d{2}-\d{2,4})"

    report = {
        'MRP': (re.search(mrp_pattern, full_blob) and ("incl" in full_blob or "tax" in full_blob), 
                "Price and 'Inclusive of all taxes' must be visible."),
        'QTY': (re.search(qty_pattern, full_blob), 
                "Standard units (g, kg, ml, l, m) not detected."),
        'DATE': (re.search(date_pattern, full_blob) or "pkd" in full_blob or "mfd" in full_blob, 
                 "Manufacturing/Packing date (MM/YYYY) missing."),
        'CARE': ("@" in full_blob or "care" in full_blob or "customer" in full_blob or re.search(r"\d{10}", full_blob), 
                 "Customer care email, phone, or address not found.")
    }
    return report

# --- APP UI ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

st.title("⚖️ Legal Metrology AI Checker")
st.write("Scan product labels for mandatory declarations (Rule 2011).")

img_file = st.camera_input("Take a photo of the product label")

if img_file:
    # Read Image
    image = Image.open(img_file)
    img_np = np.array(image)
    
    with st.spinner("Processing image for high-accuracy OCR..."):
        # Process image
        processed_img = enhance_for_ocr(img_np)
        
        # OCR
        raw_results = reader.readtext(processed_img)
        detected_text = [res[1] for res in raw_results]
        
        # Logic
        final_report = run_compliance_logic(detected_text)
        
        # Display Results
        st.subheader("Analysis Results")
        
        labels = {'MRP': 'MRP & Taxes', 'QTY': 'Net Quantity', 'DATE': 'Packing Date', 'CARE': 'Consumer Care'}
        
        for key, result in final_report.items():
            is_passed, msg = result
            status_text = "COMPLIANT (PASS)" if is_passed else "NON-COMPLIANT (FAIL)"
            status_class = "status-pass" if is_passed else "status-fail"
            
            st.markdown(f"""
                <div class="report-card">
                    <div class="card-title">{labels[key]}</div>
                    <div class="{status_class}">{status_text}</div>
                    <div class="recommendation"><b>Requirement:</b> {msg}</div>
                </div>
            """, unsafe_allow_html=True)

        with st.expander("Debug: View AI Text Extraction"):
            st.write(detected_text)
            st.image(processed_img, caption="Processed Image (What the AI reads)")
