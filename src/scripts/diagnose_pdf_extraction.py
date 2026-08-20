#!/usr/bin/env python3
"""
Standalone diagnostic: compares raw PyPDF2 vs pdfplumber extraction for
the two real sample resumes, independent of document_parser.py and the
agent pipeline, so we can see exactly what's extractable from each file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RESUME_1 = "sample_data/resumes/resume_05_khansa_aslam.pdf"
RESUME_2 = "sample_data/resumes/resume_06_ayesha_naeem.pdf"


def try_pypdf2(path):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages), len(reader.pages), None
    except Exception as e:
        return "", 0, str(e)


def try_pdfplumber(path):
    try:
        import pdfplumber
    except ImportError:
        return None, 0, "pdfplumber not installed"
    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages), len(pdf.pages), None
    except Exception as e:
        return "", 0, str(e)


def report(label, path):
    print("=" * 70)
    print(f"{label} -> {path}")
    print("=" * 70)

    text, pages, err = try_pypdf2(path)
    alpha = sum(1 for c in text if c.isalpha())
    print(f"[PyPDF2]     pages={pages}  chars={len(text)}  alpha_chars={alpha}  error={err}")
    print(f"[PyPDF2] preview: {text[:300]!r}")
    print()

    text2, pages2, err2 = try_pdfplumber(path)
    if text2 is None:
        print(f"[pdfplumber] SKIPPED - {err2}")
    else:
        alpha2 = sum(1 for c in text2 if c.isalpha())
        print(f"[pdfplumber] pages={pages2}  chars={len(text2)}  alpha_chars={alpha2}  error={err2}")
        print(f"[pdfplumber] preview: {text2[:300]!r}")
    print()


if __name__ == "__main__":
    report("Resume 05 (Khansa)", RESUME_1)
    report("Resume 06 (Ayesha)", RESUME_2)