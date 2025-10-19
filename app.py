import os
import io
import hashlib
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from fpdf import FPDF
from docx import Document as DocxDocument

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Streamlit Setup ---
st.set_page_config(page_title="🧠 AI Question Paper Generator (OCR + RAG)", page_icon="🧠", layout="wide")
st.title("🧠 AI Question Paper Generator (OCR + RAG)")
st.caption("Supports both image-based and text-based PDFs • Free stack • Caching enabled")

# --- Initialize Groq Client ---
client = Groq(api_key=GROQ_API_KEY)


# --- Utility: Cache hash for file ---
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


# --- OCR Function ---
@st.cache_data(show_spinner=False)
def extract_text_from_image_pdf(file_bytes):
    """Extract text from image-based PDFs using OCR."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(file_bytes)
        tmp_pdf.flush()

        images = convert_from_path(tmp_pdf.name)
        text_content = ""
        for page_image in images:
            # Enhance quality for OCR
            page_image = page_image.convert("L")
            page_image = page_image.filter(ImageFilter.SHARPEN)
            page_image = ImageEnhance.Contrast(page_image).enhance(2)
            text = pytesseract.image_to_string(page_image, lang="eng")
            text_content += text + "\n"

    return text_content.strip()


# --- Text Extraction ---
@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file):
    file_bytes = uploaded_file.read()
    file_hash = get_file_hash(file_bytes)

    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "".join([page.extract_text() or "" for page in reader.pages])

        # If no text found, fallback to OCR
        if not text.strip():
            st.warning(f"⚠️ No readable text found in {uploaded_file.name}. Using OCR mode...")
            text = extract_text_from_image_pdf(file_bytes)
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = ""

    if not text.strip():
        st.error(f"❌ No valid text extracted from {uploaded_file.name}. It might be blank or unreadable.")
    return text


# --- Build RAG Index ---
def build_rag_index(text_chunks, metadata):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(text_chunks, embedding=embedder, metadatas=metadata)
    return vectordb


# --- Helper: Generate Question Paper ---
def generate_question_paper(content: str, specs: dict, with_answers: bool) -> str:
    """Generate question paper using Groq API."""
    system_prompt = (
        "You are an expert exam paper generator. Based on the given material, "
        "create a question paper according to the user's specs. Keep formatting clean and realistic."
    )
    user_prompt = f"""
    Source content: {content[:3000]}
    Task:
    - Total Questions: {specs.get('total', 20)}
    - MCQs: {specs.get('mcq', 10)}, Short Answer: {specs.get('short', 5)}, Long Answer: {specs.get('long', 2)}
    - Include Answers: {"Yes" if with_answers else "No"}
    - Structure the paper with clear section titles and numbering.
    """

    try:
        response = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Groq API Error: {e}")
        return ""


# --- Utility: Save Question Paper ---
def save_question_paper(text: str, filename: str, with_answers: bool):
    os.makedirs("outputs", exist_ok=True)

    # Clean answers if requested
    if not with_answers:
        import re
        text = re.sub(r"(?i)Answer\s*[:\-].*?(?=(\nQ\d+|$))", "", text, flags=re.S)

    # --- Save as PDF ---
    pdf_path = f"outputs/{filename}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(pdf_path)

    # --- Save as DOCX ---
    docx_path = f"outputs/{filename}.docx"
    doc = DocxDocument()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(docx_path)

    return pdf_path, docx_path


# --- Streamlit UI ---
uploaded_files = st.file_uploader("📚 Upload PDF/DOCX notes", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    for f in uploaded_files:
        st.info(f"📖 Processing {f.name} ...")
        text = extract_text_from_file(f)
        if text:
            all_texts.append(text)

    if all_texts:
        full_content = "\n".join(all_texts)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_content)
        metadatas = [{"source": f"chunk-{i}"} for i in range(len(chunks))]
        vectordb = build_rag_index(chunks, metadatas)

        st.success("✅ Text extracted and indexed successfully!")

        topic = st.text_input("🎯 Which topic/chapter should be used? (e.g., Photosynthesis, Cell Division)")
        specs = {
            "total": st.number_input("Total Questions", 5, 100, 20),
            "mcq": st.number_input("MCQs", 0, 100, 10),
            "short": st.number_input("Short Questions", 0, 50, 5),
            "long": st.number_input("Long Questions", 0, 20, 2),
        }
        with_answers = st.checkbox("Include Answers?", True)

        if st.button("🧠 Generate Question Paper"):
            with st.spinner("Generating paper..."):
                question_paper = generate_question_paper(full_content, specs, with_answers)
                if question_paper:
                    pdf_path, docx_path = save_question_paper(question_paper, "AI_Question_Paper", with_answers)
                    st.success("✅ Question paper generated successfully!")
                    st.download_button("📄 Download PDF", open(pdf_path, "rb"), file_name="Question_Paper.pdf")
                    st.download_button("📝 Download DOCX", open(docx_path, "rb"), file_name="Question_Paper.docx")
