import streamlit as st
import json
import pyperclip
import os
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import easyocr
import fitz  # PyMuPDF for PDF to image conversion

# ✅ Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize Gemini LLM (LangChain wrapper)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2
)

# ✅ Initialize EasyOCR model once
reader = easyocr.Reader(['en'])  # you can add multiple languages e.g., ['en', 'hi']


# ✅ Streamlit page setup
st.set_page_config(page_title="Doc AI Agent", layout="wide")
st.title("📄 Document AI Agent")


# --- OCR with EasyOCR ---
def run_easyocr(file_bytes, file_type="image"):
    if file_type == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img_bytes = pix.tobytes("png")
            result = reader.readtext(img_bytes, detail=0, paragraph=True)
            text += "\n".join(result) + "\n"
        return text
    else:
        result = reader.readtext(file_bytes, detail=0, paragraph=True)
        return "\n".join(result)


# --- Doc Type Detection (LLM-based) ---
def detect_doc_type_llm(text: str) -> str:
    """Classify into invoice, receipt, prescription, resume, general"""
    prompt = f"""
    You are a document classifier.
    Classify the following text into one of these categories:
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
        category = response.content.strip().lower()
        if "invoice" in category:
            return "invoice"
        elif "receipt" in category:
            return "receipt"
        elif "prescription" in category:
            return "prescription"
        elif "resume" in category:
            return "resume"
        else:
            return "general"
    except Exception as e:
        st.error(f"Doc type detection failed: {e}")
        return "general"


# --- Schema per document type ---
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


# --- JSON Extraction with Retry + Confidence ---
def extract_json_with_confidence(text, schema=None, retries=2, self_consistency=False, runs=3):
    schema_hint = f"""
    Schema (fields must follow this structure):
    {json.dumps(schema, indent=2) if schema else "sensible key-value pairs"}

    - If a field is a list of objects (e.g., projects), output it as an array of JSON objects.
    - For each field, return both 'value' and 'confidence' (0.0–1.0).
    """

    prompt = f"""
    You are an AI that extracts structured data from documents.
    Document text:
    {text}

    {schema_hint}

    Output ONLY valid JSON in this format:
    {{
        "field_name": {{"value": "...", "confidence": 0.92}},
        "field_name2": {{"value": "...", "confidence": 0.75}}
    }}
    """

    def run_once(p):
        for attempt in range(retries):
            response = llm.invoke(p)
            raw_output = response.content.strip()
            cleaned_output = re.sub(r"^```(?:json)?|```$", "", raw_output, flags=re.MULTILINE).strip()
            try:
                return json.loads(cleaned_output)
            except Exception:
                if attempt == retries - 1:
                    return {"raw_output": raw_output, "error": "Invalid JSON"}
                p += "\n\n⚠️ Reminder: Output must be pure JSON, no text or markdown."

    # If self-consistency enabled, run multiple times
    if self_consistency:
        outputs = [run_once(prompt) for _ in range(runs)]
        valid = [o for o in outputs if isinstance(o, dict) and "error" not in o]
        if not valid:
            return outputs[-1]  # fallback
        # merge confidences by averaging
        merged = {}
        for o in valid:
            for field, data in o.items():
                if field not in merged:
                    merged[field] = {"value": data.get("value", ""), "confidence": data.get("confidence", 0)}
                else:
                    merged[field]["confidence"] = (merged[field]["confidence"] + data.get("confidence", 0)) / 2
        return merged
    else:
        return run_once(prompt)


# --- Validation ---
def validate_fields(result: dict):
    """Check for valid dates, numbers, emails, phones, and currency codes."""
    issues = []
    for field, data in result.items():
        if not isinstance(data, dict) or "value" not in data:
            continue
        val = str(data["value"]).strip()

        if "date" in field.lower():
            try:
                datetime.fromisoformat(val)
            except Exception:
                if not re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", val):
                    issues.append(f"⚠️ {field} has invalid date: {val}")

        if any(x in field.lower() for x in ["amount", "total", "price", "qty", "quantity"]):
            try:
                float(val.replace(",", "").replace("$", "").replace("₹", ""))
            except Exception:
                issues.append(f"⚠️ {field} is not numeric: {val}")

        if "email" in field.lower():
            if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", val):
                issues.append(f"⚠️ {field} is not a valid email: {val}")

        if "phone" in field.lower() or "contact" in field.lower():
            if not re.match(r"^\+?\d{7,15}$", val.replace(" ", "").replace("-", "")):
                issues.append(f"⚠️ {field} is not a valid phone number: {val}")

        if "currency" in field.lower():
            valid_currencies = ["USD", "INR", "EUR", "GBP", "JPY", "CAD", "AUD","$","₹","€","£","¥"]
            if val.upper() not in valid_currencies:
                issues.append(f"⚠️ {field} is not a valid currency code: {val}")
    return issues


# --- File Upload ---
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        st.info("Processing PDF ")
        text = run_easyocr(uploaded_file.read(), file_type="pdf")
    else:
        st.info("Processing Image ")
        text = run_easyocr(uploaded_file.read(), file_type="image")

    st.subheader("📜 Extracted Text")
    st.text_area("OCR Output", text, height=200)

    doc_type = detect_doc_type_llm(text)
    st.write(f"📂 Detected Document Type: **{doc_type.capitalize()}**")

    st.subheader("📝 Schema Selection")
    use_custom_schema = st.checkbox("Use custom schema?")
    if use_custom_schema:
        schema_input = st.text_area("Paste your custom schema (JSON)", height=200, value=json.dumps(SCHEMAS[doc_type], indent=2))
        try:
            custom_schema = json.loads(schema_input)
        except Exception:
            st.error("⚠️ Invalid JSON schema! Falling back to auto-detected schema.")
            custom_schema = SCHEMAS[doc_type]
    else:
        custom_schema = SCHEMAS[doc_type]

    self_consistency = st.checkbox("Enable Self-consistency (runs 3x)")

    if st.button("🔍 Extract JSON + Confidence"):
        start_time = time.time()
        result = extract_json_with_confidence(text, schema=custom_schema, self_consistency=self_consistency)
        elapsed = time.time() - start_time

        st.subheader("🗂️ Extracted JSON with Confidence")
        st.json(result)

        if isinstance(result, dict) and "error" not in result:
            st.subheader("📊 Confidence per Field")
            confidences = []
            for field, data in result.items():
                if isinstance(data, dict) and "confidence" in data:
                    conf = float(data["confidence"])
                    val = data["value"]
                    confidences.append(conf)
                    if conf < 0.6:
                        st.error(f"{field}: {val} (Low Confidence {conf:.2f})")
                    else:
                        st.write(f"**{field}**: {val} ({conf:.2f})")
                    st.progress(conf)

            if confidences:
                overall = sum(confidences) / len(confidences)
                st.info(f"📈 Overall Confidence Score: {overall:.2f}")

            issues = validate_fields(result)
            if issues:
                st.warning("⚠️ Validation Issues Found:")
                for issue in issues:
                    st.write(issue)
            else:
                st.success("✅ All fields passed validation")

            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"{doc_type}_extracted.json",
                mime="application/json"
            )

            json_str = json.dumps(result, indent=2)
            st.text_area("📜 Extracted JSON", json_str, height=200)

            if st.button("📋 Copy JSON to Clipboard"):
                pyperclip.copy(json_str)
                st.success("✅ JSON copied to clipboard!")

            if st.button("🔄 Retry Extraction"):
                st.warning("Retrying extraction...")
                st.experimental_rerun()

        st.caption(f"⏱️ Processed in {elapsed:.2f} seconds")
