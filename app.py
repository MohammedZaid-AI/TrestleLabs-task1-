import streamlit as st
import json
import pyperclip
import os
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from datetime import datetime

# ✅ Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize Gemini LLM (LangChain wrapper)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2
)

# ✅ Initialize docTR OCR model once
ocr_model = ocr_predictor(pretrained=True)

# ✅ Streamlit page setup
st.set_page_config(page_title="Doc AI Agent", layout="wide")
st.title("📄 Document AI Agent")


# --- OCR with docTR ---
def run_doctr_ocr(file_bytes, file_type="image"):
    if file_type == "pdf":
        doc = DocumentFile.from_pdf(file_bytes)
    else:
        doc = DocumentFile.from_images([file_bytes])
    result = ocr_model(doc)
    return result.render()


# --- Doc Type Detection (routing) ---
# --- Doc Type Detection (LLM-based) ---
def detect_doc_type_llm(text: str) -> str:
    """
    Use LLM to classify the document type.
    Categories: invoice, receipt, prescription, resume, general
    """
    prompt = f"""
    You are a document classifier.
    Classify the following text into one of these categories:
    - invoice
    - receipt
    - prescription
    - resume
    - general

    Document text:
    {text[:1500]}  # limit for efficiency
    """

    try:
        response = llm.invoke(prompt)
        category = response.content.strip().lower()

        # normalize
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
    "general": {
        "document_title": "string",
        "key_points": "list of strings"
    }
}


# --- JSON Extraction with Retry + Confidence ---
def extract_json_with_confidence(text, schema=None, retries=2):
    schema_hint = f"Schema: {json.dumps(schema)}" if schema else "Schema: sensible key-value pairs"

    prompt = f"""
    You are an AI that extracts structured data from documents.
    Here is the document text:

    {text}

    {schema_hint}

    For each field, return both 'value' and 'confidence' (0.0–1.0).
    Output ONLY valid JSON in this format:

    {{
        "field_name": {{"value": "...", "confidence": 0.92}},
        "field_name2": {{"value": "...", "confidence": 0.75}}
    }}
    """

    for attempt in range(retries):
        response = llm.invoke(prompt)
        raw_output = response.content.strip()

        # remove markdown fences
        cleaned_output = re.sub(r"^```(?:json)?|```$", "", raw_output, flags=re.MULTILINE).strip()

        try:
            return json.loads(cleaned_output)
        except Exception:
            if attempt == retries - 1:
                return {"raw_output": raw_output, "error": "Invalid JSON"}
            # retry with stricter instruction
            prompt += "\n\n⚠️ Reminder: Output must be pure JSON, no text or markdown."


# --- Validation ---
def validate_fields(result: dict):
    """Check for valid dates, numbers, emails, phones, and currency codes."""
    issues = []

    for field, data in result.items():
        if not isinstance(data, dict) or "value" not in data:
            continue

        val = str(data["value"]).strip()

        # --- Date Validation ---
        if "date" in field.lower():
            try:
                datetime.fromisoformat(val)
            except Exception:
                date_match = re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", val)
                if not date_match:
                    issues.append(f"⚠️ {field} has invalid date: {val}")

        # --- Numeric Validation ---
        if any(x in field.lower() for x in ["amount", "total", "price", "qty", "quantity"]):
            try:
                float(val.replace(",", "").replace("$", "").replace("₹", ""))
            except Exception:
                issues.append(f"⚠️ {field} is not numeric: {val}")

        # --- Email Validation ---
        if "email" in field.lower():
            if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", val):
                issues.append(f"⚠️ {field} is not a valid email: {val}")

        # --- Phone Number Validation ---
        if "phone" in field.lower() or "contact" in field.lower():
            if not re.match(r"^\+?\d{7,15}$", val.replace(" ", "").replace("-", "")):
                issues.append(f"⚠️ {field} is not a valid phone number: {val}")

        # --- Currency Code Validation ---
        if "currency" in field.lower():
            valid_currencies = ["USD", "INR", "EUR", "GBP", "JPY", "CAD", "AUD"]
            if val.upper() not in valid_currencies:
                issues.append(f"⚠️ {field} is not a valid currency code: {val}")

    return issues


# --- File Upload ---
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # --- OCR ---
    if uploaded_file.type == "application/pdf":
        st.info("Processing PDF with docTR...")
        text = run_doctr_ocr(uploaded_file.read(), file_type="pdf")
    else:
        st.info("Processing Image with docTR...")
        text = run_doctr_ocr(uploaded_file.read(), file_type="image")

    st.subheader("📜 Extracted Text")
    st.text_area("OCR Output", text, height=200)

    # --- Detect doc type ---
    doc_type = detect_doc_type_llm(text)
    st.write(f"📂 Detected Document Type: **{doc_type.capitalize()}**")

    # --- Allow manual schema input ---
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

    # --- LLM Extraction ---
    if st.button("🔍 Extract JSON + Confidence"):
        start_time = time.time()
        result = extract_json_with_confidence(text, schema=custom_schema)
        elapsed = time.time() - start_time

        st.subheader("🗂️ Extracted JSON with Confidence")
        st.json(result)

        # --- Validation + Confidence Bars ---
        if isinstance(result, dict) and "error" not in result:
            st.subheader("📊 Confidence per Field")
            confidences = []
            for field, data in result.items():
                if isinstance(data, dict) and "confidence" in data:
                    conf = float(data["confidence"])
                    val = data["value"]
                    confidences.append(conf)

                    # Highlight low confidence fields
                    if conf < 0.6:
                        st.error(f"{field}: {val} (Low Confidence {conf:.2f})")
                    else:
                        st.write(f"**{field}**: {val} ({conf:.2f})")
                    st.progress(conf)

            # Overall confidence score
            if confidences:
                overall = sum(confidences) / len(confidences)
                st.info(f"📈 Overall Confidence Score: {overall:.2f}")

            # Run validation
            issues = validate_fields(result)
            if issues:
                st.warning("⚠️ Validation Issues Found:")
                for issue in issues:
                    st.write(issue)
            else:
                st.success("✅ All fields passed validation")

            # Download button
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"{doc_type}_extracted.json",
                mime="application/json"
            )

            # Copy to clipboard button
            json_str = json.dumps(result, indent=2)
            st.text_area("📜 Extracted JSON", json_str, height=200)

            if st.button("📋 Copy JSON to Clipboard"):
                pyperclip.copy(json_str)
                st.success("✅ JSON copied to clipboard!")

        # --- Logging ---
        st.caption(f"⏱️ Processed in {elapsed:.2f} seconds")
