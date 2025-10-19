import streamlit as st
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from io import BytesIO
from PIL import Image
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="📘 OCR Text Extractor (Blurry PDF Support)", layout="wide")
st.title("🧠 AI Text Extractor for Blurry Scanned PDFs")
st.caption("Handles both image-based and text-based PDFs — includes preprocessing for blur & low-quality scans.")

# --- IMAGE PREPROCESSING FOR BLUR / LOW QUALITY ---
def preprocess_image(image: Image.Image):
    # Convert to OpenCV format
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Denoise and sharpen
    img = cv2.fastNlMeansDenoising(img, h=30)
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img


# --- EXTRACT TEXT FROM IMAGE ---
def extract_text_from_image(image: Image.Image):
    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed, lang="eng")
    return text


# --- EXTRACT TEXT FROM IMAGE-BASED PDF ---
@st.cache_data
def extract_text_from_image_pdf(file_bytes):
    try:
        pages = convert_from_bytes(file_bytes)
        all_text = ""
        for i, page in enumerate(pages):
            st.info(f"🖼️ Processing page {i + 1}/{len(pages)} with OCR...")
            text = extract_text_from_image(page)
            all_text += text + "\n"
        return all_text.strip()
    except pytesseract.pytesseract.TesseractNotFoundError:
        st.error("❌ Tesseract OCR not installed. Please install it locally using: `apt-get install tesseract-ocr -y`.")
        return ""
    except Exception as e:
        st.error(f"⚠️ OCR error: {e}")
        return ""


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
        st.error(f"⚠️ PDF text extraction failed: {e}")
    return text.strip()


# --- COMBINED EXTRACTION LOGIC ---
def extract_text_from_file(file_bytes):
    text = extract_text_from_text_pdf(file_bytes)
    if not text:
        st.warning("⚠️ No readable text found — switching to OCR mode (for image-based PDF).")
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
        st.error("❌ No valid text extracted. Try a clearer scan or ensure Tesseract is installed.")


st.markdown("---")
st.caption("🧩 Tip: For better OCR accuracy, upload scans with readable fonts and adequate lighting.")
