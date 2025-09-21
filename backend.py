from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage
import PyPDF2
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Root route
@app.get("/")
def root():
    return {"message": "✅ Smart ATS Backend is running"}

# Initialize Hugging Face model
hf_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=1
)
model = ChatHuggingFace(llm=hf_llm)

# PDF text extraction
def extract_pdf_text(file: UploadFile):
    import io
    reader = PyPDF2.PdfReader(io.BytesIO(file.file.read()))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# API endpoint
@app.post("/analyze_resume")
async def analyze_resume(
    resume: UploadFile = File(...),
    jd: UploadFile = File(...),
    input_prompt: str = Form(...)   # Prompt as string instead of File
):
    try:
        resume_text = extract_pdf_text(resume)
        jd_text = extract_pdf_text(jd) if jd.content_type == "application/pdf" else jd.file.read().decode("utf-8")
        
        final_prompt = input_prompt.replace("{resume}", resume_text).replace("{jd}", jd_text)

        response = model([HumanMessage(content=final_prompt)])

        try:
            ats_result = json.loads(response.content)
            return ats_result
        except json.JSONDecodeError:
            return {"raw_output": response.content}

    except Exception as e:
        return {"error": str(e)}