📄 Document AI Agent

An AI-powered Document Intelligence app that extracts structured JSON from PDFs, images, and handwritten documents with confidence scoring, validation, and a clean UI.

📂 Supported Document Types

🧾 Invoices
🧾 Receipts
💊 Prescriptions
👨‍💼 Resumes
📄 General documents

⚡ Features

OCR with EasyOCR → Handles PDFs, images, and handwritten text.

Document Type Detection (LLM-based) → Auto-classifies docs into invoice, receipt, prescription, resume, or general.

Structured JSON Extraction → Converts messy text into clean JSON aligned with schemas.

Confidence Scoring → Per-field + overall score, low-confidence values flagged.

Validation → Dates, numerics, emails, phone numbers, and currency codes auto-validated.

UI/UX →

JSON viewer

Confidence bars

Low-confidence warnings

Download JSON

Copy-to-clipboard

Custom Schema Support → Paste your own schema to guide extraction.

🛠️ Tech Stack

Python 3.9+

Streamlit – Web UI

EasyOCR + PyMuPDF (fitz) – OCR for images & PDFs

LangChain + Google Gemini API – LLM-powered extraction

Pyperclip – Copy JSON to clipboard

Regex + Python built-ins – Validation

🚀 Setup & Run

Clone repo

git clone https://github.com/MohammedZaid-AI/TrestleLabs-task1-.git
cd document-ai-agent


Create virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Set environment variables
Create a .env file with:

GOOGLE_API_KEY=your_google_api_key


Run the app

streamlit run app.py

📊 Example Output

Input – Handwritten Receipt
<img src="examples/handwritten_receipt.jpg" width="400"/>

Output – Extracted JSON

{
  "store_name": {"value": "Bright Caterers", "confidence": 0.60},
  "date": {"value": "2016-07-01", "confidence": 0.50},
  "items": {"value": ["Catering service"], "confidence": 0.90},
  "total_amount": {"value": 5500.0, "confidence": 0.90}
}

🧩 Solution Approach

OCR (EasyOCR + PyMuPDF) extracts raw text from PDFs/images.

LLM (Gemini via LangChain) classifies the document type.

Schema-guided extraction → prompts LLM to return JSON with value + confidence.

Validation ensures extracted fields make sense (dates, numerics, emails, currency).

Confidence scoring highlights reliability of extracted fields.

UI (Streamlit) presents results with confidence bars, warnings, and JSON export.

📉 Limitations & Trade-offs

Handwritten text → EasyOCR works but confidence drops on messy handwriting.

Domain-specific fields → Schema may need extensions (e.g., prescription dosage/timing).

LLM cost/latency → Dependent on API usage and document size.

Currency codes → Still need normalization ($ → USD).

Dates → Ambiguous formats may cause validation issues.

📅 Evaluation Criteria Mapping

✔️ Confidence Quality (40%) → Per-field + overall scoring, low-confidence flagged.
✔️ Extraction Accuracy (25%) → Correct schema fields, validated numerics/dates.
✔️ Modeling & Prompting (10%) → Concise schema-guided prompts, routing logic.
✔️ Engineering Robustness (10%) → Retries, schema checks, latency logs.
✔️ UI/UX (10%) → Clean JSON viewer, confidence bars, download/copy.
✔️ Documentation (5%) → README with setup, examples, limitations.

📂 Example test files are provided in the samples/ folder for quick evaluation (digital invoices, handwritten receipts, and resumes).