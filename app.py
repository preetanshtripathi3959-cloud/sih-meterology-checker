import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import re

# --- SETTING UP THE PAGE ---
st.set_page_config(page_title="Legal Metrology AI", page_icon="⚖️", layout="centered")

# --- UI STYLING ---
st.markdown("""
    <style>
    .report-card { padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin-bottom: 10px; }
    .pass { color: #28a745; font-weight: bold; }
    .fail { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_ocr_model():
    # Downloads the model on first run
    return easyocr.Reader(['en'])

reader = load_ocr_model()

# --- LEGAL METROLOGY LOGIC ENGINE ---
def analyze_compliance(text_lines):
    full_text = " ".join(text_lines).lower()
    
    # Rules definitions based on Packaged Commodities Rules 2011
    checks = {
        "MRP Declaration": {
            "passed": bool(re.search(r"(mrp|rs|retail).*?\d+", full_text)) and ("inclusive" in full_text or "incl" in full_text),
            "requirement": "Must show MRP and 'inclusive of all taxes'.",
            "found": re.findall(r"(?:mrp|rs\.?)\s?(\d+(?:\.\d+)?)", full_text)
        },
        "Net Quantity": {
            "passed": bool(re.search(r"(\d+\.?\d*)\s?(g|kg|ml|l|unit|n|pcs|qty)", full_text)),
            "requirement": "Standard units (g, kg, ml, l) must be visible.",
            "found": re.findall(r"(\d+\.?\d*)\s?(g|kg|ml|l|unit|n|pcs|qty)", full_text)
        },
        "Month & Year of Pkd": {
            "passed": bool(re.search(r"(\d{2}/\d{4}|\d{2}-\d{2,4})", full_text)) or "pkd" in full_text or "mfd" in full_text,
            "requirement": "Must declare Month and Year of manufacture/packing.",
            "found": re.findall(r"(\d{2}/\d{4}|\d{2}-\d{2,4})", full_text)
        },
        "Consumer Care Details": {
            "passed": "@" in full_text or bool(re.search(r"(\d{10}|1800)", full_text)) or "consumer" in full_text,
            "requirement": "Contact email or phone number is mandatory.",
            "found": "Contact info detected" if "@" in full_text else "Missing"
        }
    }
    return checks

# --- MAIN WEB INTERFACE ---
st.title("🛡️ Legal Metrology Compliance AI")
st.subheader("Smart India Hackathon 2024 Prototype")
st.write("Scan any product label to check if it follows the **Legal Metrology (Packaged Commodities) Rules, 2011**.")

# Step 1: Camera Input (Works on Mobile/Laptop)
img_file = st.camera_input("Take a clear photo of the product label")

if img_file:
    # Convert image for processing
    image = Image.open(img_file)
    img_np = np.array(image)
    
    with st.spinner("AI is analyzing label compliance..."):
        # Step 2: Run OCR
        results = reader.readtext(img_np)
        detected_text = [res[1] for res in results]
        
        # Step 3: Run Logic Engine
        report = analyze_compliance(detected_text)
        
        # Step 4: Display Results
        st.divider()
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        
        # Calculate overall compliance percentage
        passed_count = sum(1 for item in report.values() if item["passed"])
        score = (passed_count / len(report)) * 100
        
        st.metric("Compliance Score", f"{int(score)}%")
        
        st.write("### Detailed Report:")
        for label, data in report.items():
            status_class = "pass" if data["passed"] else "fail"
            status_icon = "✅" if data["passed"] else "❌"
            
            with st.container():
                st.markdown(f"""
                <div class="report-card">
                    <span class="{status_class}">{status_icon} {label}</span><br>
                    <small><b>Rule:</b> {data['requirement']}</small><br>
                    <small><b>Detected:</b> {data['found'] if data['found'] else 'None'}</small>
                </div>
                """, unsafe_allow_html=True)

        # Show raw OCR output for judges to see transparency
        with st.expander("View AI Text Extraction (Debugging)"):
            st.write(detected_text)

# --- FOOTER ---
st.divider()
st.caption("Developed for SIH Problem Statement: Automated Compliance for Packaged Commodities.")