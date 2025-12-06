import streamlit as st
import json
import pyperclip
import os
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

import pytesseract
from PIL import Image
import fitz  # PyMuPDF PDF parsing
import io
import shutil


# ============================================================
# 0️⃣ LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# ============================================================
# 1️⃣ SETUP GEMINI LLM
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2
)


# ============================================================
# 2️⃣ TESSERACT CONFIGURATION (Windows + Linux)
# ============================================================

def configure_tesseract():
    """
    Handles OCR path setup for:
    - Windows (your correct installed path)
    - Linux / HuggingFace Spaces (auto-detect)
    """

    if os.name == "nt":
        # Windows — using your detected correct install paths
        tesseract_exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        tessdata_dir = r"C:\Program Files\Tesseract-OCR\tessdata"

        if not os.path.exists(tesseract_exe):
            raise FileNotFoundError("❌ Tesseract not found at expected Windows location.")

        if not os.path.exists(os.path.join(tessdata_dir, "eng.traineddata")):
            raise FileNotFoundError("❌ Missing eng.traineddata in tessdata folder.")

        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
        os.environ["TESSDATA_PREFIX"] = tessdata_dir

        print("✔ Windows Tesseract configured:", tesseract_exe)
        print("✔ Using tessdata:", tessdata_dir)

    else:
        # Linux / HuggingFace Spaces
        linux_path = shutil.which("tesseract")
        if linux_path:
            pytesseract.pytesseract.tesseract_cmd = linux_path
            # Standard HF tessdata location
            os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
            print("✔ Linux Tesseract configured:", linux_path)
        else:
            raise FileNotFoundError("❌ Tesseract not found on this Linux environment.")


# Run configuration
configure_tesseract()


# ============================================================
# 3️⃣ OCR ENGINE — TESSERACT
# ============================================================

def run_tesseract(file_bytes, file_type="image"):
    """
    Extract text using Tesseract OCR.
    Handles both PDF and Image files.
    """

    text_out = ""

    # PDF case
    if file_type == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            extracted = pytesseract.image_to_string(img)
            text_out += extracted + "\n"

        return text_out.strip()

    # Image case
    img = Image.open(io.BytesIO(file_bytes))
    extracted = pytesseract.image_to_string(img)

    return extracted.strip()


# ============================================================
# 4️⃣ DOCUMENT CLASSIFICATION (LLM)
# ============================================================

def detect_doc_type_llm(text: str) -> str:
    prompt = f"""
    Classify this document into one category:
    - invoice
    - receipt
    - prescription
    - resume
    - general

    Document text:
    {text[:1500]}
    """

    try:
        response = llm.invoke(prompt)
        label = response.content.strip().lower()

        for t in ["invoice", "receipt", "prescription", "resume"]:
            if t in label:
                return t
        return "general"

    except:
        return "general"


# ============================================================
# 5️⃣ SCHEMAS FOR EXTRACTION
# ============================================================

SCHEMAS = {
    "invoice": {
        "invoice_number": "string",
        "date": "YYYY-MM-DD",
        "total_amount": "float",
        "currency": "string",
        "vendor_name": "string"
    },
    "prescription": {
        "patient_name": "string",
        "doctor_name": "string",
        "date": "YYYY-MM-DD",
        "medications": "list of strings"
    },
    "receipt": {
        "store_name": "string",
        "date": "YYYY-MM-DD",
        "items": "list of item names",
        "total_amount": "float"
    },
    "resume": {
        "name": "string",
        "email": "string",
        "phone": "string",
        "education": "list of {degree, institution, year}",
        "experience": "list of {company, role, duration}",
        "projects": "list of {title, description, duration}",
        "skills": "list of strings"
    },
    "general": {
        "document_title": "string",
        "key_points": "list of strings"
    }
}


# ============================================================
# 6️⃣ JSON EXTRACTION (LLM + CONFIDENCE)
# ============================================================

def extract_json_with_confidence(text, schema, retries=2, self_consistency=False, runs=3):

    schema_hint = json.dumps(schema, indent=2)

    prompt = f"""
    Extract structured JSON from the document:

    Document:
    {text}

    Schema:
    {schema_hint}

    Return ONLY valid JSON.
    Each field must be:
    {{
        "value": "...",
        "confidence": 0.0–1.0
    }}
    """

    def run_once(p):
        for attempt in range(retries):
            response = llm.invoke(p)
            raw = response.content.strip()

            cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

            try:
                return json.loads(cleaned)
            except:
                if attempt == retries - 1:
                    return {"error": "Invalid JSON", "raw_output": raw}

                p += "\nREMINDER: Output MUST be valid JSON only."

    if self_consistency:
        outputs = [run_once(prompt) for _ in range(runs)]
        valid = [o for o in outputs if "error" not in o]

        if not valid:
            return outputs[-1]

        merged = {}
        for o in valid:
            for field, d in o.items():
                if field not in merged:
                    merged[field] = {"value": d["value"], "confidence": d["confidence"]}
                else:
                    merged[field]["confidence"] = (merged[field]["confidence"] + d["confidence"]) / 2

        return merged

    return run_once(prompt)


# ============================================================
# 7️⃣ FIELD VALIDATION ENGINE
# ============================================================

def validate_fields(result):
    issues = []

    for field, data in result.items():
        if not isinstance(data, dict):
            continue

        val = str(data.get("value", "")).strip()

        # Date validation
        if "date" in field.lower():
            try:
                datetime.fromisoformat(val)
            except:
                if not re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", val):
                    issues.append(f"⚠️ Invalid date: {field} → {val}")

        # Numeric fields
        if any(x in field.lower() for x in ["amount", "total", "price"]):
            try:
                float(val.replace(",", "").replace("$", "").replace("₹", ""))
            except:
                issues.append(f"⚠️ Invalid number: {field} → {val}")

        # Email
        if "email" in field.lower():
            if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", val):
                issues.append(f"⚠️ Invalid email format → {val}")

    return issues


# ============================================================
# 8️⃣ STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Doc AI Agent", layout="wide")
st.title("📄 Document AI Agent (Tesseract OCR + Gemini JSON Extractor)")


uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:

    # OCR processing
    if uploaded_file.type == "application/pdf":
        st.info("📄 Running OCR on PDF using Tesseract…")
        text = run_tesseract(uploaded_file.read(), "pdf")
    else:
        st.info("🖼️ Running OCR on Image using Tesseract…")
        text = run_tesseract(uploaded_file.read(), "image")

    st.subheader("📜 OCR Extracted Text")
    st.text_area("Extracted Text", text, height=250)

    # Document type
    doc_type = detect_doc_type_llm(text)
    st.write(f"📂 Detected Type: **{doc_type}**")

    # Schema selection
    st.subheader("📑 Schema Settings")
    custom = st.checkbox("Use Custom Schema")

    if custom:
        schema_text = st.text_area("Custom Schema", json.dumps(SCHEMAS[doc_type], indent=2), height=200)
        try:
            schema = json.loads(schema_text)
        except:
            st.error("Invalid JSON schema! Using default.")
            schema = SCHEMAS[doc_type]
    else:
        schema = SCHEMAS[doc_type]

    sc = st.checkbox("Enable Self-Consistency (3 runs)")

    if st.button("🔍 Extract JSON"):
        start = time.time()

        result = extract_json_with_confidence(text, schema, self_consistency=sc)

        st.subheader("🗂️ Extracted JSON")
        st.json(result)

        # Validation
        issues = validate_fields(result)
        if issues:
            st.warning("⚠️ Validation issues found:")
            for i in issues:
                st.write(i)
        else:
            st.success("✔ All fields validated successfully")

        # Download JSON
        json_str = json.dumps(result, indent=2)
        st.download_button("💾 Download JSON", json_str, file_name=f"{doc_type}_data.json")

        st.text_area("Final JSON", json_str, height=200)

        if st.button("📋 Copy JSON"):
            pyperclip.copy(json_str)
            st.success("Copied to clipboard!")

        st.caption(f"⏱️ Completed in {time.time() - start:.2f} seconds")
