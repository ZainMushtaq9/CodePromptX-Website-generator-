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

# LangChain + embeddings + vectorstore (FREE & CLOUD-SAFE)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------- Utilities ----------------------
def sanitize_for_latin(text: str) -> str:
    """Ensure FPDF-compatible text (latin-1)."""
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


def chunk_documents_from_uploads(uploaded_files) -> Tuple[List[str], List[dict]]:
    """Extract and chunk text from uploaded PDFs/DOCXs."""
    texts, metadatas = [], []
    for file in uploaded_files:
        name = file.name
        temp_path = os.path.join(tempfile.gettempdir(), name)
        with open(temp_path, "wb") as f:
            f.write(file.read())

        if name.lower().endswith(".pdf"):
            reader = PdfReader(temp_path)
            for i, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                texts.append(txt)
                metadatas.append({"source": name, "page": i})
        elif name.lower().endswith(".docx"):
            doc = DocxReader(temp_path)
            txt = "\n".join(p.text for p in doc.paragraphs)
            texts.append(txt)
            metadatas.append({"source": name, "page": 0})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, chunk_metas = [], []
    for t, m in zip(texts, metadatas):
        parts = splitter.split_text(t)
        for idx, p in enumerate(parts):
            chunks.append(p)
            chunk_metas.append({**m, "chunk": idx})
    return chunks, chunk_metas


# ---------------------- RAG (Free Embedding + Vector DB) ----------------------
def build_rag_index(chunks: List[str], metadatas: List[dict]):
    """Build a Chroma vector index (in-memory, fully free)."""
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embedding=embedder, metadatas=metadatas)
    return vectordb


def retrieve_context(vectordb, keywords: str = None, top_k: int = 6) -> str:
    """Retrieve relevant chunks by keyword query."""
    if keywords and keywords.strip():
        docs = vectordb.similarity_search(keywords, k=top_k)
    else:
        docs = vectordb.similarity_search("", k=top_k)
    return "\n\n".join([d.page_content for d in docs])


# ---------------------- Question Paper Generator ----------------------
def generate_question_paper(content: str, specs: dict, with_answers: bool) -> str:
    """Generate question paper using Groq compound-mini."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    system_prompt = (
        "You are an expert academic question paper generator. Create a formatted paper "
        "with MCQs, Short Answers, and Long Answers. Add numbering and structure. "
        "If with_answers=True, include answer keys at the end; otherwise omit them."
    )

    safe_content = content[:18000]
    user_prompt = (
        f"Context:\n{safe_content}\n\n"
        f"Specs:\nTotal: {specs.get('total')}, MCQs: {specs.get('mcq')}, "
        f"Short: {specs.get('short')}, Long: {specs.get('long')}, "
        f"With Answers: {with_answers}\n\n"
        "Generate the question paper clearly and concisely."
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


# ---------------------- Save Question Paper ----------------------
def save_question_paper(text: str, filename: str, with_answers: bool):
    """Save output as DOCX and PDF."""
    os.makedirs("outputs", exist_ok=True)
    content = re.sub(r"\r\n", "\n", text).strip()
    if not with_answers:
        content = re.split(r"(?i)\nanswers?\b", content)[0]

    # DOCX
    docx_path = os.path.join("outputs", f"{filename}.docx")
    doc = DocxWriter()
    doc.add_heading(filename.title(), 0)
    for p in content.split("\n"):
        doc.add_paragraph(p)
    doc.save(docx_path)

    # PDF
    pdf_path = os.path.join("outputs", f"{filename}.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in content.split("\n"):
        pdf.multi_cell(0, 7, sanitize_for_latin(line))
    pdf.output(pdf_path)


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="AI Question Paper Generator", page_icon="🧠")
st.title("🧠 AI Question Paper Generator (Groq + RAG)")
st.caption("Free Groq + HuggingFace + Chroma Stack — 100% Streamlit compatible.")

uploaded = st.file_uploader("Upload your PDFs/DOCX", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded and st.button("Build Knowledge Base"):
    with st.spinner("Building vector index..."):
        chunks, metas = chunk_documents_from_uploads(uploaded)
        vectordb = build_rag_index(chunks, metas)
        st.session_state["vectordb"] = vectordb
        st.success("✅ Knowledge base ready!")

if "vectordb" in st.session_state:
    st.subheader("Generate Question Paper")
    topic = st.text_input("Enter topic, chapter, or keywords (optional):")
    total_q = st.number_input("Total Questions", 1, 100, 20)
    mcq_q = st.number_input("MCQs", 0, 100, 10)
    short_q = st.number_input("Short Answers", 0, 100, 5)
    long_q = st.number_input("Long Answers", 0, 100, 2)
    with_answers = st.checkbox("Include Answers", True)
    filename = st.text_input("File name", "question_paper")

    if st.button("Generate 🚀"):
        with st.spinner("Retrieving and generating..."):
            context = retrieve_context(st.session_state["vectordb"], topic, top_k=6)
            specs = {"total": total_q, "mcq": mcq_q, "short": short_q, "long": long_q}
            result = generate_question_paper(context, specs, with_answers)
            if "Error" in result:
                st.error(result)
            else:
                save_question_paper(result, filename, with_answers)
                st.success("✅ Generated successfully! Download below:")
                with open(f"outputs/{filename}.pdf", "rb") as fpdf:
                    st.download_button("📥 Download PDF", fpdf, file_name=f"{filename}.pdf")
                with open(f"outputs/{filename}.docx", "rb") as fdocx:
                    st.download_button("📥 Download DOCX", fdocx, file_name=f"{filename}.docx")
