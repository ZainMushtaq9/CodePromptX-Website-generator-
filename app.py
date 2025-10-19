import os
import re
import tempfile
import unicodedata
from typing import List, Tuple

import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document as DocxReader
from fpdf import FPDF
from docx import Document as DocxWriter

# Free LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# OCR
import pytesseract
from pdf2image import convert_from_path


# ---------------------- Utilities ----------------------
def sanitize_for_latin(text: str) -> str:
    """Remove or replace characters that FPDF can't handle (latin-1 safe)."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------- PDF & DOC Extraction ----------------------
def extract_text_from_pdf(path: str) -> List[str]:
    """
    Extract text from both text-based and image-based PDFs.
    For image pages, uses OCR with pytesseract.
    """
    texts = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, start=1):
            txt = page.extract_text()
            if txt and len(txt.strip()) > 30:
                texts.append(txt.strip())
            else:
                # Fallback: OCR
                try:
                    images = convert_from_path(path, first_page=i, last_page=i)
                    for im in images:
                        ocr_txt = pytesseract.image_to_string(im)
                        if len(ocr_txt.strip()) > 30:
                            texts.append(ocr_txt.strip())
                except Exception:
                    continue
    except Exception:
        return []
    return [t for t in texts if len(t.strip()) > 30]


def chunk_documents_from_uploads(uploaded_files) -> Tuple[List[str], List[dict]]:
    """Extract and chunk text from PDF/DOCX uploads (OCR aware)."""
    texts, metadatas = [], []

    for file in uploaded_files:
        name = file.name
        temp_path = os.path.join(tempfile.gettempdir(), name)
        with open(temp_path, "wb") as f:
            f.write(file.read())

        if name.lower().endswith(".pdf"):
            pdf_texts = extract_text_from_pdf(temp_path)
            if not pdf_texts:
                st.warning(f"⚠️ No readable text found in {name}. It might be blank or low-quality scan.")
                continue
            for i, t in enumerate(pdf_texts):
                texts.append(t)
                metadatas.append({"source": name, "page": i})

        elif name.lower().endswith(".docx"):
            try:
                doc = DocxReader(temp_path)
                full_text = "\n".join([p.text for p in doc.paragraphs])
                if len(full_text.strip()) < 30:
                    st.warning(f"⚠️ {name} seems empty or unreadable.")
                    continue
                texts.append(full_text)
                metadatas.append({"source": name, "page": 0})
            except Exception:
                st.warning(f"❌ Error reading {name}. Skipping.")
                continue

    if not texts:
        st.error("❌ No valid text extracted from uploads.")
        return [], []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, metas = [], []
    for t, m in zip(texts, metadatas):
        for i, part in enumerate(splitter.split_text(t)):
            if len(part.strip()) > 20:
                chunks.append(part)
                metas.append({**m, "chunk": i})

    return chunks, metas


# ---------------------- RAG Index ----------------------
def build_rag_index(chunks: List[str], metadatas: List[dict]):
    """Build Chroma vector index safely (no empty embeddings)."""
    if not chunks:
        raise ValueError("No text chunks to index.")
    chunks = [c for c in chunks if len(c.strip()) > 0]

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vectordb = Chroma.from_texts(chunks, embedding=embedder, metadatas=metadatas)
    except ValueError as e:
        if "Expected Embeddings" in str(e):
            raise ValueError("No valid text found for embeddings. Try re-uploading a clearer document.")
        else:
            raise e

    return vectordb


def retrieve_context(vectorstore, keywords=None, by_pages=None, top_k=6):
    """Retrieve relevant chunks by keywords or page range."""
    results = []
    if by_pages:
        s, e = by_pages
        docs = vectorstore.similarity_search("", k=top_k * 5)
        for d in docs:
            p = d.metadata.get("page", 0)
            if s <= p <= e:
                results.append(d.page_content)
            if len(results) >= top_k:
                break
    elif keywords:
        docs = vectorstore.similarity_search(keywords, k=top_k)
        results = [d.page_content for d in docs]
    else:
        docs = vectorstore.similarity_search("", k=top_k)
        results = [d.page_content for d in docs]
    return "\n\n".join(results)


# ---------------------- Groq Model ----------------------
def generate_question_paper(content: str, specs: dict, with_answers: bool) -> str:
    """Use Groq to generate question paper."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system_prompt = (
        "You are an expert teacher. Generate a structured question paper from the study material below. "
        "Include three sections: MCQs, Short Answer, Long Answer. "
        "If with_answers=True, include an answer key at the end."
    )
    content = content[:18000]
    user_prompt = (
        f"Study material:\n{content}\n\n"
        f"Specs:\nTotal: {specs['total']} | MCQ: {specs['mcq']} | Short: {specs['short']} | Long: {specs['long']} | "
        f"Include Answers: {with_answers}"
    )

    try:
        resp = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating paper: {e}"


# ---------------------- Save Utility ----------------------
def save_question_paper(text: str, filename: str, with_answers: bool):
    os.makedirs("outputs", exist_ok=True)
    content = re.sub(r"\r\n", "\n", text).strip()
    if not with_answers:
        content = re.split(r"(?i)\nanswers?\b", content)[0]

    # DOCX
    docx_path = os.path.join("outputs", f"{filename}.docx")
    doc = DocxWriter()
    doc.add_heading(filename.title(), 0)
    for para in content.split("\n"):
        doc.add_paragraph(para)
    doc.save(docx_path)

    # PDF
    pdf_path = os.path.join("outputs", f"{filename}.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in content.split("\n"):
        pdf.multi_cell(0, 7, sanitize_for_latin(line))
    pdf.output(pdf_path)

    return docx_path, pdf_path


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="🧠 AI Question Paper Generator (OCR + RAG)", page_icon="🧠")
st.title("🧠 AI Question Paper Generator (OCR + RAG)")
st.caption("Supports both text & image-based PDFs • 100% free stack")

uploaded = st.file_uploader("📚 Upload PDF or DOCX notes", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded and st.button("🔎 Build Knowledge Index"):
    with st.spinner("Extracting & indexing... please wait."):
        chunks, metas = chunk_documents_from_uploads(uploaded)
        if chunks:
            try:
                vectordb = build_rag_index(chunks, metas)
                st.session_state["vectordb"] = vectordb
                st.success(f"✅ Indexed {len(chunks)} chunks of text.")
            except Exception as e:
                st.error(f"❌ Index build error: {e}")

# Retrieval
st.subheader("🎯 Choose your topic or pages")
colA, colB = st.columns(2)
with colA:
    keywords = st.text_input("Search by keywords (e.g., Photosynthesis, Algebra, etc.)", "")
with colB:
    page_range = st.text_input("Or specify page range (e.g., 5-10)", "")

top_k = st.slider("Context size (chunks)", 2, 10, 6)

# Question Specs
st.subheader("📄 Question Paper Settings")
total = st.number_input("Total Questions", 1, 100, 20)
mcq = st.number_input("MCQs", 0, 50, 10)
short = st.number_input("Short Answer", 0, 20, 5)
long = st.number_input("Long Answer", 0, 10, 2)
with_answers = st.checkbox("Include Answers", True)
filename = st.text_input("Output file name", "question_paper")

if st.button("🚀 Generate Question Paper"):
    if "vectordb" not in st.session_state:
        st.warning("Please build the RAG index first.")
    else:
        vectordb = st.session_state["vectordb"]
        pages_tuple = None
        if page_range and re.match(r"^\d+\s*-\s*\d+$", page_range):
            s, e = [int(x) for x in page_range.split("-")]
            pages_tuple = (min(s, e), max(s, e))
        with st.spinner("Generating question paper..."):
            context = retrieve_context(vectordb, keywords=keywords, by_pages=pages_tuple, top_k=top_k)
            if not context.strip():
                st.error("❌ No relevant context found.")
            else:
                specs = {"total": total, "mcq": mcq, "short": short, "long": long}
                result = generate_question_paper(context, specs, with_answers)
                if result.startswith("Error"):
                    st.error(result)
                else:
                    docx_path, pdf_path = save_question_paper(result, filename, with_answers)
                    st.success("✅ Question paper generated!")
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF", f, file_name=f"{filename}.pdf")
                    with open(docx_path, "rb") as f:
                        st.download_button("📥 Download DOCX", f, file_name=f"{filename}.docx")

st.info("Workflow → 1️⃣ Upload notes → 2️⃣ Build index → 3️⃣ Choose topic/pages → 4️⃣ Generate paper")
