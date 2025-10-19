import os
import io
import re
import tempfile
import unicodedata

import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document as DocxReader
from docx import Document as DocxWriter
from fpdf import FPDF

# LangChain & Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# OCR
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="🧠 AI Question Paper Generator (OCR + RAG)", page_icon="🧠")
st.title("🧠 AI Question Paper Generator (OCR + RAG)")
st.caption("Supports both image-based and text-based PDFs • Free stack • Smart caching")

# ---------------------- UTILITIES ----------------------
def sanitize_for_latin(text: str) -> str:
    """Ensure text can be encoded in latin-1 for FPDF."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    encoded = text.encode("latin-1", errors="replace")
    return encoded.decode("latin-1")


@st.cache_data(show_spinner=False)
def extract_text_from_text_pdf(file_bytes: bytes) -> str:
    """Extract text from a text-based PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        reader = PdfReader(tmp.name)
        text = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
        return text.strip()


@st.cache_data(show_spinner=False)
def extract_text_from_image_pdf(file_bytes: bytes) -> str:
    """Extract text from scanned (image-based) PDFs using PyMuPDF + Tesseract."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(file_bytes)
        tmp_pdf.flush()

        doc = fitz.open(tmp_pdf.name)
        for page_number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(image)
            if page_text.strip():
                text += f"\n\n[Page {page_number}]\n{page_text.strip()}"
        doc.close()
    return text.strip()


@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file):
    """Handles PDF (text or scanned) and DOCX files."""
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text = extract_text_from_text_pdf(file_bytes)
        if not text.strip():
            st.warning(f"⚠️ No readable text found in {uploaded_file.name}. Using OCR mode...")
            text = extract_text_from_image_pdf(file_bytes)
    elif name.endswith(".docx"):
        doc = DocxReader(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = ""
    return text


def build_rag_index(chunks, metadatas):
    """Build vectorstore (Chroma) from chunks."""
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embedding=embedder, metadatas=metadatas)
    return vectordb


def chunk_text(text, source):
    """Split text into retrievable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
    return chunks, metadatas


def generate_question_paper(content, specs, with_answers=True):
    """Generate a question paper using Groq free API."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system_prompt = (
        "You are an expert academic exam paper generator. "
        "Create a formatted paper with MCQs, Short, and Long Questions. "
        "If 'with_answers' is True, add an answer key at the end."
    )
    user_prompt = (
        f"Study Material:\n{content[:15000]}\n\n"
        f"Specs: Total={specs['total']}, MCQ={specs['mcq']}, Short={specs['short']}, Long={specs['long']}, "
        f"Include Answers={with_answers}"
    )
    try:
        resp = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating question paper: {e}"


def save_outputs(content, filename, with_answers=True):
    """Save generated paper as PDF and DOCX."""
    os.makedirs("outputs", exist_ok=True)
    if not with_answers:
        content = re.split(r"(?i)\nanswers?\b", content)[0]

    docx_path = os.path.join("outputs", f"{filename}.docx")
    pdf_path = os.path.join("outputs", f"{filename}.pdf")

    # DOCX
    doc = DocxWriter()
    doc.add_heading(filename.title(), 1)
    for para in content.split("\n"):
        doc.add_paragraph(para)
    doc.save(docx_path)

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in content.split("\n"):
        safe_line = sanitize_for_latin(line)
        pdf.multi_cell(0, 7, safe_line)
    pdf.output(pdf_path)

    return docx_path, pdf_path

# ---------------------- STREAMLIT APP ----------------------
uploaded = st.file_uploader("📚 Upload PDF/DOCX notes", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded:
    all_texts, all_metas = [], []
    for f in uploaded:
        st.write(f"📖 Processing {f.name} ...")
        text = extract_text_from_file(f)
        if not text.strip():
            st.error(f"❌ No readable text extracted from {f.name}.")
            continue
        chunks, metas = chunk_text(text, f.name)
        all_texts.extend(chunks)
        all_metas.extend(metas)

    if all_texts:
        vectordb = build_rag_index(all_texts, all_metas)
        st.session_state["vectordb"] = vectordb
        st.success("✅ Index built successfully!")

# Question Paper Options
st.subheader("📝 Question Paper Settings")
total_q = st.number_input("Total Questions", 1, 100, 20)
mcq_q = st.number_input("MCQs", 0, 50, 10)
short_q = st.number_input("Short Questions", 0, 20, 5)
long_q = st.number_input("Long Questions", 0, 20, 2)
with_answers = st.checkbox("Include Answers", True)
filename = st.text_input("Output filename", "question_paper")

if st.button("🚀 Generate Question Paper"):
    if "vectordb" not in st.session_state:
        st.warning("Please upload and index your notes first.")
    else:
        vectordb = st.session_state["vectordb"]
        docs = vectordb.similarity_search("", k=6)
        context = "\n\n".join([d.page_content for d in docs])
        specs = {"total": total_q, "mcq": mcq_q, "short": short_q, "long": long_q}
        result = generate_question_paper(context, specs, with_answers)
        if "Error" in result:
            st.error(result)
        else:
            docx_path, pdf_path = save_outputs(result, filename, with_answers)
            st.success("✅ Generated successfully!")
            with open(pdf_path, "rb") as fpdf:
                st.download_button("📥 Download PDF", fpdf, file_name=f"{filename}.pdf")
            with open(docx_path, "rb") as fdocx:
                st.download_button("📥 Download DOCX", fdocx, file_name=f"{filename}.docx")
