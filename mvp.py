import os, re, sys, json
import fitz  # PyMuPDF
import pdfplumber
import docx
from docx import Document
from langdetect import detect
from PIL import Image
import pytesseract

import smtp  # your smtp.py module

# Summariser
from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

OUTPUT_DIR = "snapshots_out"

STAKEHOLDER_EMAILS = {
    "Engineering": ["arshitsood5@gmail.com"],
    "Finance": ["mtyagi2002@gmail.com"],
    "Safety": ["purvikabansal05@gmail.com"],
    "HR": ["saanyasetia@gmail.com"],
    "Management": ["gandhinavnidhi@gmail.com"],
}


# ---------------- Helpers ----------------
def detect_language(text: str) -> str:
    try: return detect(text)
    except: return "unknown"

def normalize_spaces(text: str) -> str:
    text = re.sub(r'-\s+\n', '', text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 3000):
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 1 <= max_chars:
            current += ("\n" + p if current else p)
        else:
            chunks.append(current)
            current = p
    if current: chunks.append(current)
    return chunks

def _ocr_page(page) -> str:
    pix = page.get_pixmap()
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    txt = pytesseract.image_to_string(img, lang="eng")  # add +mal if needed
    return normalize_spaces(txt)


# ---------------- Extraction ----------------
def extract_pdf(file_path: str):
    out = []
    doc = fitz.open(file_path)

    for i in range(len(doc)):
        page = doc[i]

        # --- Text ---
        raw = page.get_text("rawdict") # type: ignore
        lines = []
        for block in raw.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    spans = line.get("spans", [])
                    txt = "".join([s.get("text", "") for s in spans if "text" in s])
                    if txt.strip(): lines.append(txt)
        full_text = normalize_spaces("\n".join(lines))

        # OCR fallback if little/no text
        if len(full_text) < 25:
            full_text = _ocr_page(page)

        # --- Tables ---
        tables_text = ""
        with pdfplumber.open(file_path) as pdf:
            try:
                tables = pdf.pages[i].extract_tables()
                for t in tables:
                    rows = [" | ".join([c.strip() if c else "" for c in row]) for row in t]
                    tables_text += "\n[TABLE]\n" + "\n".join(rows)
            except: pass

        # --- Figures ---
        images = page.get_images(full=True)
        figs_text = ""
        for img_index, img in enumerate(images):
            figs_text += f"\n[FIGURE: page {i+1}, image_{img_index+1}]"

        content = "\n".join([full_text, tables_text, figs_text]).strip()
        out.append({
            "doc_id": os.path.basename(file_path),
            "page": i+1,
            "type": "pdf",
            "lang": detect_language(content) if content else "unknown",
            "content": content
        })
    doc.close()
    return out


def extract_docx(file_path: str):
    doc = docx.Document(file_path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    text = normalize_spaces("\n".join(paras))
    return [{
        "doc_id": os.path.basename(file_path),
        "page": 1,
        "type": "docx",
        "lang": detect_language(text) if text else "unknown",
        "content": text
    }]


def extract_eml(file_path: str):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    subject = re.search(r"Subject:(.*)", data)
    subject_text = subject.group(1).strip() if subject else ""
    body = re.split(r"\n\n", data, maxsplit=1)[-1]
    text = f"Subject: {subject_text}\n\n{body}"
    return [{
        "doc_id": os.path.basename(file_path),
        "page": 1,
        "type": "email",
        "lang": detect_language(text) if text else "unknown",
        "content": normalize_spaces(text)
    }]


def extract_any(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf": return extract_pdf(file_path)
    if ext == ".docx": return extract_docx(file_path)
    if ext == ".eml": return extract_eml(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------- Summarisation ----------------
def summarize_with_ai(text: str) -> str:
    if not text.strip(): return "No content extracted."
    chunks = chunk_text(text)
    results = []
    for ch in chunks:
        try:
            res = summarizer(ch, max_length=200, min_length=60, do_sample=False, truncation=True)
            results.append(res[0]["summary_text"])
        except Exception:
            results.append(ch[:300])  # fallback: raw excerpt
    return "\n".join(results)


# ---------------- Snapshots ----------------
def make_snapshot(doc_id: str, stakeholder: str, summary: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"{os.path.splitext(os.path.basename(doc_id))[0]}__{stakeholder}_snapshot.docx"
    path = os.path.join(OUTPUT_DIR, fname)

    d = Document()
    d.add_heading(f"{stakeholder} Snapshot", 0)
    d.add_paragraph(f"Source: {os.path.basename(doc_id)}")
    d.add_paragraph("")

    d.add_heading("Summary", level=1)
    for para in summary.split("\n"):
        if para.strip(): d.add_paragraph(para.strip())

    d.add_paragraph("")
    d.add_paragraph(f"Traceability: {os.path.basename(doc_id)}")
    d.save(path)
    return path


# ---------------- Pipeline ----------------
def process_file(file_path: str):
    elements = extract_any(file_path)
    all_text = "\n\n".join([e["content"] for e in elements if e["content"].strip()])
    summary = summarize_with_ai(all_text)

    outputs = []
    for role in ["Engineering", "Finance", "Safety", "HR", "Management"]:
        docx_path = make_snapshot(file_path, role, summary)
        outputs.append((role, docx_path))

        # Send via your smtp.py
        smtp.send_email(
            subject=f"[Snapshot] {os.path.basename(file_path)} — {role}",
            body="Please find the attached role-specific snapshot.",
            recipients=STAKEHOLDER_EMAILS.get(role, []),
            attachments=[docx_path]
        )
    return outputs


# ---------------- CLI ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kmrl_phase1_ai.py <file.pdf|.docx|.eml>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    out = process_file(input_path)
    print("\n✅ Snapshots generated and emailed:")
    for role, path in out:
        print(f" - {role}: {path}")
