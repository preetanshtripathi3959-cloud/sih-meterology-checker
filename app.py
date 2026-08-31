import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import re
from rapidfuzz import process, fuzz

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Legal Metrology AI Explorer", layout="wide", page_icon="⚖️")

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004085; color: white; }
    .report-card {
        background-color: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 8px solid #004085;
        margin-bottom: 20px;
    }
    .status-pass { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .status-fail { color: #dc3545; font-weight: bold; font-size: 1.2em; }
    .metric-box { text-align: center; padding: 10px; background: #e9ecef; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- LANGUAGE DICTIONARY ---
LANG_DATA = {
    "English": {
        "header": "⚖️ Legal Metrology Compliance AI",
        "desc": "Automated Check for Packaged Commodities Rules, 2011",
        "sidebar_title": "Control Panel",
        "lang_select": "Choose Language / भाषा चुनें",
        "cam_btn": "Scan Product Label",
        "ana_btn": "Run Analysis",
        "results": "Compliance Report",
        "mrp_label": "MRP & Tax Declaration",
        "qty_label": "Net Quantity Check",
        "date_label": "Date of Packing (Mfg)",
        "care_label": "Consumer Care Info",
        "pass": "COMPLIANT (PASS)",
        "fail": "NON-COMPLIANT (FAIL)",
        "suggestion": "Recommendation"
    },
    "Hindi (हिन्दी)": {
        "header": "⚖️ कानूनी मेट्रोलॉजी एआई",
        "desc": "पैकेज्ड कमोडिटीज रूल्स, 2011 के लिए स्वचालित जांच",
        "sidebar_title": "नियंत्रण कक्ष",
        "lang_select": "भाषा चुनें",
        "cam_btn": "लेबल स्कैन करें",
        "ana_btn": "विश्लेषण शुरू करें",
        "results": "अनुपालन रिपोर्ट",
        "mrp_label": "MRP और कर घोषणा",
        "qty_label": "शुद्ध मात्रा की जांच",
        "date_label": "पैकिंग की तारीख",
        "care_label": "उपभोक्ता देखभाल जानकारी",
        "pass": "अनुपालन (पास)",
        "fail": "गैर-अनुपालन (फेल)",
        "suggestion": "सुझाव"
    }
}

# --- ACCURACY BOOSTER: IMAGE PRE-PROCESSING ---
def enhance_image(img_array):
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Increase contrast (Denoising)
    dst = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Binary Thresholding to make text pop
    _, thr = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr

# --- COMPLIANCE ENGINE WITH FUZZY MATCHING ---
def run_compliance_logic(text_list):
    full_blob = " ".join(text_list).lower()
    
    # Legal Keywords to find (handles OCR typos)
    mrp_keywords = ["mrp", "retail price", "maximum retail", "incl of all taxes"]
    qty_keywords = ["net quantity", "net weight", "net qty", "volume"]
    
    report = {}

    # 1. MRP Check
    mrp_found = any(fuzz.partial_ratio(kw, full_blob) > 80 for kw in mrp_keywords)
    has_tax_phrase = "inclusive" in full_blob or "incl" in full_blob
    report['MRP'] = (mrp_found and has_tax_phrase, "Price & 'Inclusive of all taxes' phrase required.")

    # 2. Quantity Check
    qty_found = re.search(r'(\d+\.?\d*)\s?(g|kg|ml|l|unit|n|pcs)', full_blob)
    report['QTY'] = (bool(qty_found), "Must use standard units (g, kg, ml, l).")

    # 3. Date Check (MM/YYYY)
    date_found = re.search(r'(\d{2}/\d{4})|(\d{2}/\d{2})', full_blob)
    report['DATE'] = (bool(date_found), "Month and Year of packing is mandatory.")

    # 4. Consumer Care Check
    care_found = "@" in full_blob or "consumer" in full_blob or bool(re.search(r'\d{10}', full_blob))
    report['CARE'] = (care_found, "Email or Phone number of consumer care must be present.")

    return report

# --- MAIN APP ---
@st.cache_resource
def get_reader():
    return easyocr.Reader(['en', 'hi']) # English + Hindi OCR

reader = get_reader()

# Sidebar Setup
with st.sidebar:
    st.title(LANG_DATA["English"]["sidebar_title"])
    selected_lang = st.selectbox("Language", ["English", "Hindi (हिन्दी)"])
    L = LANG_DATA[selected_lang]
    st.divider()
    st.markdown("**SIH 2024 - Team Prototype**")

st.title(L["header"])
st.write(L["desc"])

# Camera/Upload Section
img_file = st.camera_input(L["cam_btn"])

if img_file:
    # Processing
    image = Image.open(img_file)
    img_np = np.array(image)
    
    # Accuracy Boost: Enhance Image
    processed_img = enhance_image(img_np)
    
    with st.spinner("AI analyzing compliance metrics..."):
        # Run OCR on processed image
        raw_results = reader.readtext(processed_img)
        detected_text = [res[1] for res in raw_results]
        
        # Run Compliance Logic
        final_report = run_compliance_logic(detected_text)
        
        # Layout Results
        st.subheader(L["results"])
        
        # Dashboard Score
        score = sum(1 for v in final_report.values() if v[0])
        st.progress(score / 4)
        
        col1, col2 = st.columns(2)
        
        # Map logic to UI labels
        ui_mapping = {
            'MRP': L['mrp_label'],
            'QTY': L['qty_label'],
            'DATE': L['date_label'],
            'CARE': L['care_label']
        }

        for key, result in final_report.items():
            is_passed, msg = result
            status_text = L["pass"] if is_passed else L["fail"]
            status_class = "status-pass" if is_passed else "status-fail"
            
            with st.container():
                st.markdown(f"""
                <div class="report-card">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-size:1.3em; font-weight:bold;">{ui_mapping[key]}</span>
                        <span class="{status_class}">{status_text}</span>
                    </div>
                    <hr style="margin:10px 0;">
                    <b>{L['suggestion']}:</b> {msg}
                </div>
                """, unsafe_allow_html=True)

        # Show Debugging info for Judges
        with st.expander("Show AI Raw Extraction"):
            st.write(detected_text)
            st.image(processed_img, caption="What the AI sees (Pre-processed)")
