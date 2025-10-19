import streamlit as st
from groq import Groq
import json
import zipfile
import io

# --- Streamlit Page Setup ---
st.set_page_config(page_title="CodePromptX - WebApp Generator", page_icon="⚡")
st.title("⚡ CodePromptX — Full WebApp Generator")
st.caption("Free AI Project Builder — Powered by GroqCloud (Model: groq/compound-mini)")

# --- Initialize Groq client ---
client = Groq(api_key=st.secrets.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY"))

# --- User Inputs ---
st.subheader("🧠 Describe your project idea")
project_goal = st.text_area("What do you want to build?", placeholder="e.g. A blog platform with React frontend and Flask backend")

col1, col2, col3 = st.columns(3)
with col1:
    frontend = st.selectbox("Frontend", ["React", "Next.js", "Streamlit", "HTML/CSS/JS"])
with col2:
    backend = st.selectbox("Backend", ["Flask", "FastAPI", "Node.js", "Django", "None"])
with col3:
    database = st.selectbox("Database", ["MongoDB", "PostgreSQL", "MySQL", "SQLite", "None"])

# --- Generate Button ---
if st.button("🚀 Generate My WebApp"):
    if not project_goal.strip():
        st.warning("Please describe your project idea first.")
    else:
        with st.spinner("⚙️ Generating your project... Please wait..."):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional full-stack project generator. "
                            "Given a tech stack and purpose, generate a working directory structure "
                            "with clear file contents and commands to run backend and frontend separately. "
                            "Respond in valid JSON with keys: directories, files, run_instructions."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Project Goal: {project_goal}
                        Frontend: {frontend}
                        Backend: {backend}
                        Database: {database}
                        Return a JSON output with directory structure and code.
                        """
                    }
                ]

                response = client.chat.completions.create(
                    model="groq/compound-mini",
                    messages=messages,
                    temperature=0.7,
                )

                output = response.choices[0].message.content

                # Try parsing the output as JSON
                try:
                    project_data = json.loads(output)
                except json.JSONDecodeError:
                    st.error("⚠️ AI output not in perfect JSON — showing raw text instead:")
                    st.code(output, language="json")
                    st.stop()

                # Display structure and run instructions
                st.subheader("📂 Generated Project Structure")
                st.json(project_data.get("directories", {}))

                st.subheader("💻 Run Instructions")
                for step in project_data.get("run_instructions", []):
                    st.markdown(f"- `{step}`")

                # --- Create ZIP file for download ---
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for folder, files in project_data.get("directories", {}).items():
                        for filename, content in files.items():
                            zip_file.writestr(f"{folder}/{filename}", content)
                zip_buffer.seek(0)

                st.download_button(
                    label="⬇️ Download Project ZIP",
                    data=zip_buffer,
                    file_name="generated_webapp.zip",
                    mime="application/zip",
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")

st.divider()
st.caption("💡 Example prompt: 'Generate a Flask + React + MongoDB app for managing notes.'")
