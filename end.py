import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage
import PyPDF2
from dotenv import load_dotenv
import json
import sqlite3
from datetime import datetime
import pandas as pd
import os
import re

# Load environment variables
load_dotenv()
# frontend.py


# ---- HuggingFace Model Setup ----
hf_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=1,
    huggingfacehub_api_token=os.getenv("HF_API_KEY")
)
model = ChatHuggingFace(llm=hf_llm)

# ---- SQLite Setup ----
conn = sqlite3.connect("ats_results.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS ats_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_filename TEXT,
    jd_filename TEXT,
    resume_text TEXT,
    jd_text TEXT,
    result_json TEXT,
    created_at TEXT
)
''')
conn.commit()

def save_ats_result(resume_file, jd_file, resume_text, jd_text, result_dict):
    c.execute('''
    INSERT INTO ats_results (resume_filename, jd_filename, resume_text, jd_text, result_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        resume_file.name,
        jd_file.name,
        resume_text,
        jd_text,
        json.dumps(result_dict),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

def fetch_ats_history():
    c.execute("SELECT id, resume_filename, jd_filename, result_json, created_at FROM ats_results ORDER BY created_at DESC")
    return c.fetchall()

# ---- Helpers ----
def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    if not text.strip():
        return "[⚠️ Could not extract text, maybe it's scanned PDF.]"
    return text

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_pdf_text(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return ""

def safe_json_parse(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        st.error(f"JSON parsing failed: {e}")
        return None

# ---- Weighted Scoring ----
def calculate_weighted_score(ats_result, resume_text, jd_text):
    """
    Hybrid scoring: 70% LLM score + 30% keyword overlap.
    """
    try:
        llm_score = int(ats_result.get("Relevance Score", 0))
    except:
        llm_score = 0

    jd_tokens = set(re.findall(r"\b\w+\b", jd_text.lower()))
    resume_tokens = set(re.findall(r"\b\w+\b", resume_text.lower()))

    keyword_overlap = (len(jd_tokens & resume_tokens) / max(1, len(jd_tokens))) * 100
    final_score = round(0.7 * llm_score + 0.3 * keyword_overlap, 2)
    return final_score

# ---- Prompt Template ----
input_prompt = """
You are an Automated Resume Relevance Check System.
Compare the student's resume with the given job description (JD) and provide structured ATS-style feedback.

Resume: {resume}
Job Description: {jd}

Output MUST be JSON in this format:
{{
  "Relevance Score": "<0-100>",
  "Suitability": "<High/Medium/Low>",
  "Matched Skills": [list],
  "Missing Skills": [list],
  "Missing Certifications": [list],
  "Missing Projects": [list],
  "Suggestions": [
    "Actionable improvement points."
  ]
}}
"""

# ---- Streamlit UI ----
st.set_page_config(page_title="Smart ATS", layout="wide")
st.title("📄 Smart ATS: Resume Evaluation System")
st.write("Upload multiple resumes and a Job Description to get automated ATS-style evaluation at scale.")

resume_files = st.file_uploader("Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)
jd_file = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

if st.button("Submit"):
    if not resume_files or not jd_file:
        st.error("⚠️ Please upload resumes and a job description before submitting.")
    else:
        with st.spinner("Analyzing resumes..."):
            jd_text = extract_text(jd_file)
            results = []

            for resume_file in resume_files:
                resume_text = extract_text(resume_file)
                final_prompt = input_prompt.format(resume=resume_text, jd=jd_text)

                response = model([HumanMessage(content=final_prompt)])
                response_text = response.content.strip()

                ats_result = safe_json_parse(response_text)
                if ats_result:
                    # Compute weighted score
                    final_score = calculate_weighted_score(ats_result, resume_text, jd_text)
                    ats_result["Final Score"] = final_score

                    # Save in DB
                    save_ats_result(resume_file, jd_file, resume_text, jd_text, ats_result)

                    # Collect for batch display
                    results.append({
                        "Resume": resume_file.name,
                        "Relevance Score": ats_result.get("Relevance Score", "N/A"),
                        "Final Score": ats_result.get("Final Score", "N/A"),
                        "Suitability": ats_result.get("Suitability", "N/A"),
                        "Matched Skills": ", ".join(ats_result.get("Matched Skills", [])),
                        "Missing Skills": ", ".join(ats_result.get("Missing Skills", [])),
                        "Missing Certifications": ", ".join(ats_result.get("Missing Certifications", [])),
                        "Missing Projects": ", ".join(ats_result.get("Missing Projects", [])),
                        "Suggestions": " | ".join(ats_result.get("Suggestions", [])),
                    })
            
            if results:
                df = pd.DataFrame(results)
                st.markdown("### 🏆 ATS Evaluation Results")
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "📥 Download Results (CSV)",
                    df.to_csv(index=False),
                    file_name="ats_results.csv",
                    mime="text/csv"
                )

                st.success("✅ Evaluations saved to database.")

# ---- Past Evaluations Dashboard ----
st.markdown("---")
st.header("📜 Past ATS Evaluations")

history = fetch_ats_history()

if history:
    for record in history:
        record_id, resume_name, jd_name, result_json, created_at = record
        result_dict = json.loads(result_json)

        with st.expander(f"Resume: {resume_name} | JD: {jd_name} | Date: {created_at}"):
            st.metric("Relevance Score", result_dict.get("Relevance Score", "N/A"))
            st.metric("Final Score", result_dict.get("Final Score", "N/A"))
            st.metric("Suitability", result_dict.get("Suitability", "N/A"))

            st.markdown("**Matched Skills:**")
            for skill in result_dict.get("Matched Skills", []):
                st.markdown(f"- {skill}")

            st.markdown("**Missing Skills:**")
            for skill in result_dict.get("Missing Skills", []):
                st.markdown(f"- {skill}")

            st.markdown("**Missing Certifications:**")
            for cert in result_dict.get("Missing Certifications", []):
                st.markdown(f"- {cert}")

            st.markdown("**Missing Projects:**")
            for proj in result_dict.get("Missing Projects", []):
                st.markdown(f"- {proj}")

            st.markdown("**Suggestions:**")
            for point in result_dict.get("Suggestions", []):
                st.markdown(f"- {point}")
else:
    st.info("No past evaluations found.")

# ---- Recruiter Dashboard ----
st.markdown("---")
st.header("📊 Recruiter Dashboard: ATS Insights")

history = fetch_ats_history()

if history:
    records = []
    for record in history:
        record_id, resume_name, jd_name, result_json, created_at = record
        result_dict = json.loads(result_json)
        records.append({
            "Resume": resume_name,
            "Job Description": jd_name,
            "Relevance Score": int(result_dict.get("Relevance Score", 0)),
            "Final Score": float(result_dict.get("Final Score", 0)),
            "Suitability": result_dict.get("Suitability", "N/A"),
            "Matched Skills": ", ".join(result_dict.get("Matched Skills", [])),
            "Missing Skills": ", ".join(result_dict.get("Missing Skills", [])),
            "Missing Certifications": ", ".join(result_dict.get("Missing Certifications", [])),
            "Missing Projects": ", ".join(result_dict.get("Missing Projects", [])),
            "Suggestions": " | ".join(result_dict.get("Suggestions", [])),
            "Created At": created_at
        })
    
    df = pd.DataFrame(records)

    # Filters
    st.subheader("🔍 Filter Results")
    suitability_filter = st.multiselect("Suitability", df["Suitability"].unique(), default=df["Suitability"].unique())
    min_score, max_score = st.slider("Final Score Range", 0, 100, (0, 100))

    filtered_df = df[
        (df["Suitability"].isin(suitability_filter)) &
        (df["Final Score"].between(min_score, max_score))
    ]

    st.dataframe(filtered_df, use_container_width=True)

    # Metrics
    st.subheader("📈 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Evaluations", len(df))
    col2.metric("High Suitability", (df["Suitability"] == "High").sum())
    col3.metric("Avg Final Score", round(df["Final Score"].mean(), 2))

    # Chart: Suitability distribution
    st.subheader("📊 Suitability Distribution")
    suitability_counts = df["Suitability"].value_counts()
    st.bar_chart(suitability_counts)

    # Export
    st.download_button(
        "📥 Download Dashboard Data (CSV)",
        filtered_df.to_csv(index=False),
        file_name="ats_dashboard.csv",
        mime="text/csv"
    )
else:
    st.info("No past evaluations found.")

