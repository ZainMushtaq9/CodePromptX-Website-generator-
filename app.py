import streamlit as st
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
import shutil

st.set_page_config(page_title="🧠 AI Text Extractor (OCR + Smart Fallbacks)", layout="wide")
st.title("🧠 AI Text Extractor for Blurry Scanned PDFs")
st.caption("Handles both image-based and text-based PDFs — with smart fallbacks for missing system libraries.")


# --- UTILITY: Check system dependencies ---
def is_tool_installed(tool_name: str):
    return shutil.which(tool_name) is not None


# --- IMAGE PREPROCESSING ---
def preprocess_image(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.fastNlMeansDenoising(img, h=25)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


# --- OCR EXTRACTION ---
def extract_text_from_image(image: Image.Image):
    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed, lang="eng")
    return text


# --- EXTRACT TEXT FROM TEXT-BASED PDF ---
@st.cache_data
def extract_text_from_text_pdf(file_bytes):
    text = ""
    try:
        with BytesIO(file_bytes) as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Text-based PDF extraction failed: {e}")
    return text.strip()


# --- EXTRACT TEXT FROM IMAGE-BASED PDF ---
@st.cache_data
def extract_text_from_image_pdf(file_bytes):
    if not is_tool_installed("pdftoppm"):
        st.error("❌ Poppler (pdftoppm) is not installed. OCR cannot run. Please install it locally.")
        st.info("👉 On Linux: `sudo apt-get install poppler-utils -y`\n👉 On Windows: install from https://blog.alivate.com.au/poppler-windows/")
        return ""

    if not is_tool_installed("tesseract"):
        st.error("❌ Tesseract OCR not installed. Please install it locally.")
        st.info("👉 On Windows: install from https://github.com/UB-Mannheim/tesseract/wiki\n👉 On Linux: `sudo apt-get install tesseract-ocr -y`")
        return ""

    try:
        pages = convert_from_bytes(file_bytes)
        all_text = ""
        for i, page in enumerate(pages):
            st.info(f"🖼️ Processing page {i + 1}/{len(pages)} via OCR...")
            text = extract_text_from_image(page)
            all_text += text + "\n"
        return all_text.strip()

    except Exception as e:
        st.error(f"⚠️ OCR extraction error: {e}")
        return ""


# --- MAIN EXTRACTION LOGIC ---
def extract_text_from_file(file_bytes):
    text = extract_text_from_text_pdf(file_bytes)
    if not text:
        st.warning("⚠️ No readable text found — switching to OCR mode (for scanned PDFs).")
        text = extract_text_from_image_pdf(file_bytes)
    return text


# --- UI ---
uploaded_file = st.file_uploader("📚 Upload your PDF", type=["pdf"])

if uploaded_file:
    st.info(f"Processing: {uploaded_file.name}")
    with st.spinner("🔍 Extracting text..."):
        text = extract_text_from_file(uploaded_file.read())

    if text:
        st.success("✅ Text extracted successfully!")
        st.text_area("📜 Extracted Text", text, height=400)
    else:
        st.error("❌ No valid text extracted. Try a clearer scan or ensure Poppler & Tesseract are installed.")
