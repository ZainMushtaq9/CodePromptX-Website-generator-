import os
import io
import re
import tempfile
import unicodedata
from typing import List, Dict, Tuple

import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document as DocxReader
from fpdf import FPDF
from docx import Document as DocxWriter
from PIL import Image
import pytesseract

# LangChain + embeddings + vectorstore (free + cloud-safe)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------- Utilities ----------------------
def sanitize_for_latin(text: str) -> str:
    """Ensure text safe for FPDF."""
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

# ---------------------- Document Processing ----------------------
def extract_text_from_pdf(path: str) -> List[str]:
    """
    Extract text from both text-based and image-based PDFs.
    For image pages, use pytesseract OCR.
    Returns list of per-page text.
    """
    texts = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt and len(txt.strip()) > 30:
                texts.append(txt)
            else:
                # OCR fallback for scanned page
                images = page.images
                if not images:
                    # convert page to image via pdf2image (optional lightweight fallback)
                    from pdf2image import convert_from_path
                    temp_images = convert_from_path(path, first_page=i + 1, last_page=i + 1)
                    for im in temp_images:
                        ocr_text = pytesseract.image_to_string(im)
                        texts.append(ocr_text)
                else:
                    for im in images:
                        img_bytes = io.BytesIO(im.data)
                        image = Image.open(img_bytes)
                        ocr_text = pytesseract.image_to_string(image)
                        texts.append(ocr_text)
    except Exception:
        return []
    return texts


def chunk_documents_from_uploads(uploaded_files) -> Tuple[List[str], List[dict]]:
    """Extract and chunk text from PDF/DOCX (OCR-aware)."""
    texts, metadatas = [], []
    for file in uploaded_files:
        name = file.name
        temp_path = os.path.join(tempfile.gettempdir(), name)
        with open(temp_path, "wb") as f:
            f.write(file.read())

        if name.lower().endswith(".pdf"):
            pdf_texts = extract_text_from_pdf(temp_path)
            for i, t in enumerate(pdf_texts):
                texts.append(t)
                metadatas.append({"source": name, "page": i + 1})
        elif name.lower().endswith(".docx"):
            doc = DocxReader(temp_path)
            full = "\n".join([p.text for p in doc.paragraphs])
            texts.append(full)
            metadatas.append({"source": name, "page": 0})
        else:
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, chunk_metas = [], []
    for t, m in zip(texts, metadatas):
        for i, part in enumerate(splitter.split_text(t)):
            chunks.append(part)
            chunk_metas.append({**m, "chunk": i})
    return chunks, chunk_metas

# ---------------------- RAG ----------------------
def build_rag_index(chunks: List[str], metadatas: List[dict]):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embedding=embedder, metadatas=metadatas)
    return vectordb

def retrieve_context(vectordb, keywords: str = None, top_k: int = 6) -> str:
    docs = vectordb.similarity_search(keywords or "", k=top_k)
    return "\n\n".join([d.page_content for d in docs])

# ---------------------- Groq Question Generator ----------------------
def generate_question_paper(content: str, specs: dict, with_answers: bool) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system_prompt = (
        "You are an expert teacher assistant. Generate a question paper "
        "with MCQs, Short, and Long answers. Include 'Answers' section if requested."
    )
    user_prompt = (
        f"Context:\n{content[:18000]}\n\n"
        f"Specs:\nTotal: {specs['total']}, MCQ: {specs['mcq']}, Short: {specs['short']}, Long: {specs['long']}, "
        f"With Answers: {with_answers}\n"
        "Generate the question paper accordingly."
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
        return f"Error generating question paper: {e}"

# ---------------------- Save Outputs ----------------------
def save_question_paper(text: str, filename: str, with_answers: bool):
    os.makedirs("outputs", exist_ok=True)
    if not with_answers:
        text = re.split(r"(?i)\nanswers?\b", text)[0]

    # DOCX
    doc = DocxWriter()
    doc.add_heading(filename.title(), 0)
    for para in text.split("\n"):
        doc.add_paragraph(para)
    doc.save(f"outputs/{filename}.docx")

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in text.split("\n"):
        pdf.multi_cell(0, 7, sanitize_for_latin(line))
    pdf.output(f"outputs/{filename}.pdf")

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="AI Question Paper Generator", page_icon="🧠")
st.title("🧠 AI Question Paper Generator (OCR + RAG)")
st.caption("Supports image-based and text-based PDFs • Free stack")

uploaded = st.file_uploader("📚 Upload PDF/DOCX notes", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded and st.button("🔎 Build Knowledge Base"):
    with st.spinner("Extracting & indexing text..."):
        chunks, metas = chunk_documents_from_uploads(uploaded)
        vectordb = build_rag_index(chunks, metas)
        st.session_state["vectordb"] = vectordb
        st.success("✅ Knowledge base built successfully!")

if "vectordb" in st.session_state:
    st.subheader("🧩 Generate Question Paper")
    topic = st.text_input("Enter topic or keywords (optional):")
    total_q = st.number_input("Total Questions", 1, 100, 20)
    mcq_q = st.number_input("MCQs", 0, 100, 10)
    short_q = st.number_input("Short Answers", 0, 100, 5)
    long_q = st.number_input("Long Answers", 0, 100, 2)
    with_answers = st.checkbox("Include Answers", True)
    filename = st.text_input("File name", "question_paper")

    if st.button("🚀 Generate"):
        with st.spinner("Generating paper..."):
            context = retrieve_context(st.session_state["vectordb"], topic, top_k=6)
            specs = {"total": total_q, "mcq": mcq_q, "short": short_q, "long": long_q}
            result = generate_question_paper(context, specs, with_answers)
            if "Error" in result:
                st.error(result)
            else:
                save_question_paper(result, filename, with_answers)
                st.success("✅ Generated successfully!")
                with open(f"outputs/{filename}.pdf", "rb") as fpdf:
                    st.download_button("📥 Download PDF", fpdf, file_name=f"{filename}.pdf")
                with open(f"outputs/{filename}.docx", "rb") as fdocx:
                    st.download_button("📥 Download DOCX", fdocx, file_name=f"{filename}.docx")
