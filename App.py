import os
import json
import tempfile
import zipfile
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_keys = os.getenv("GROQ_API_KEYS", "").split(",")

# Function to rotate through multiple API keys
def create_client():
    for key in api_keys:
        try:
            client = Groq(api_key=key.strip())
            # test connection (light ping)
            client.chat.completions.create(
                model="groq/compound-mini",
                messages=[{"role": "system", "content": "ping"}]
            )
            return client
        except Exception:
            continue
    st.error("All API keys failed. Please check your keys.")
    return None

# Generate project structure and code using AI
def generate_code_structure(language, framework, db=None):
    client = create_client()
    if not client:
        return None

    prompt = f"""
You are an expert full-stack developer.
Generate a complete project structure for a {language} project using {framework}.
- Include folders and files with starter code and README.
- Follow best practices for {language}/{framework}.
- Optional database: {db}.
- Output ONLY in JSON format:
{{
  "folders": ["list of folders"],
  "files": {{
      "filename": "file content"
  }},
  "instructions": "how to run the project"
}}
"""
    response = client.chat.completions.create(
        model="groq/compound-mini",
        messages=[
            {"role": "system", "content": "You are a professional code generator AI."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Create zip file from JSON data
def create_zip_from_json(json_data):
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        st.error("Failed to parse JSON from AI. Try again.")
        return None

    tmp_dir = tempfile.mkdtemp()

    # Create folders
    for folder in data.get("folders", []):
        os.makedirs(os.path.join(tmp_dir, folder), exist_ok=True)

    # Create files
    for fname, content in data.get("files", {}).items():
        file_path = os.path.join(tmp_dir, fname)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # Create zip
    zip_path = os.path.join(tempfile.gettempdir(), "project.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    arcname=os.path.relpath(os.path.join(root, file), tmp_dir)
                )
    return zip_path

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="AI Code Generator", page_icon="💻")
st.title("💻 CodePromptX — AI Code Generator")
st.caption("Generate full project structure with starter code using Groq (Free Model: compound-mini)")

# User selections
language = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "Go", "C#"])
framework = st.selectbox("Framework / Backend", ["Flask", "FastAPI", "Django", "React", "Node.js", "Spring Boot"])
db = st.selectbox("Database (Optional)", ["None", "SQLite", "PostgreSQL", "MySQL", "MongoDB"])

# Generate button
if st.button("🚀 Generate Project"):
    with st.spinner("Generating project..."):
        json_result = generate_code_structure(language, framework, db if db != "None" else None)
        if json_result:
            zip_file = create_zip_from_json(json_result)
            if zip_file:
                st.success("Project generated successfully! 🎉")
                st.download_button(
                    label="⬇️ Download Project.zip",
                    data=open(zip_file, "rb"),
                    file_name="project.zip",
                    mime="application/zip"
                )

st.divider()
st.caption("💡 Tip: Choose your language and framework, and AI will generate a ready-to-run project for you.")
