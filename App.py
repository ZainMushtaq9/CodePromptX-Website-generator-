import os
import streamlit as st
from groq import Groq
import tempfile
from PyPDF2 import PdfReader
from docx import Document
from fpdf import FPDF
from docx import Document as DocxDocument
import re

# ---------------------- Helper Function ----------------------
def generate_question_paper(content: str, specs: dict, with_answers: bool) -> str:
    """
    Calls Groq compound-mini model to generate a question paper.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    system_prompt = (
        "You are an expert academic exam paper generator. "
        "Given the provided text (study material), create a complete question paper. "
        "Follow the structure strictly according to the requested number of questions: "
        "MCQs, short answers, and long answers. "
        "If 'with_answers' is False, do not include any solutions or answer keys."
    )
    user_prompt = (
        f"Generate a question paper based on this material:\n\n{content[:6000]}\n\n"
        f"Specifications:\nTotal Questions: {specs['total']}\n"
        f"MCQs: {specs['mcq']}\nShort Answer: {specs['short']}\nLong Answer: {specs['long']}\n"
        f"Include Answers: {with_answers}\n\n"
        "Format neatly with clear question numbering and sections."
    )
    try:
        response = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating question paper: {e}"


# ---------------------- Utility Function ----------------------
def save_question_paper(text: str, filename: str, with_answers: bool) -> None:
    """
    Cleans and formats text into PDF and DOCX question papers.
    """
    os.makedirs("outputs", exist_ok=True)
    clean_text = text

    if not with_answers:
        clean_text = re.sub(r"(?i)(Answer:.*?)(?=\n|$)", "", clean_text)
        clean_text = re.sub(r"(?i)(Explanation:.*?)(?=\n|$)", "", clean_text)

    # --- Save PDF ---
    pdf_path = os.path.join("outputs", f"{filename}.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in clean_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(pdf_path)

    # --- Save DOCX ---
    docx_path = os.path.join("outputs", f"{filename}.docx")
    doc = DocxDocument()
    doc.add_heading("Generated Question Paper", level=1)
    for line in clean_text.split("\n"):
        doc.add_paragraph(line)
    doc.save(docx_path)


# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="AI Question Paper Generator", page_icon="📝")
st.title("📝 AI Question Paper Generator")
st.caption("Powered by Groq Cloud — Model: compound-mini")

uploaded_files = st.file_uploader("📚 Upload your textbook or notes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
total_q = st.number_input("Total Questions", min_value=1, value=20)
mcq_q = st.number_input("Number of MCQs", min_value=0, value=10)
short_q = st.number_input("Short Answer Questions", min_value=0, value=5)
long_q = st.number_input("Long Answer Questions", min_value=0, value=2)
with_answers = st.checkbox("Include Answers", value=True)

if st.button("🚀 Generate Question Paper"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Generating your question paper... please wait."):
            try:
                full_text = ""
                for file in uploaded_files:
                    temp_path = os.path.join(tempfile.gettempdir(), file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.read())

                    if file.name.lower().endswith(".pdf"):
                        reader = PdfReader(temp_path)
                        for page in reader.pages:
                            full_text += page.extract_text() + "\n"
                    elif file.name.lower().endswith(".docx"):
                        doc = Document(temp_path)
                        for para in doc.paragraphs:
                            full_text += para.text + "\n"

                specs = {"total": total_q, "mcq": mcq_q, "short": short_q, "long": long_q}
                output = generate_question_paper(full_text, specs, with_answers)

                if output.startswith("Error"):
                    st.error(output)
                else:
                    save_question_paper(output, "question_paper", with_answers)
                    st.success("✅ Question paper generated successfully!")

                    with open("outputs/question_paper.pdf", "rb") as pdf_file:
                        st.download_button("📥 Download PDF", pdf_file, file_name="question_paper.pdf")

                    with open("outputs/question_paper.docx", "rb") as docx_file:
                        st.download_button("📥 Download Word", docx_file, file_name="question_paper.docx")

            except Exception as e:
                st.error(f"❌ Error: {e}")

st.divider()
st.caption("💡 Example: Upload your notes and request 20 MCQs, 5 short, and 2 long questions with answers.")
