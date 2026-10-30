"""PDF Text and Candidate Information Extractor module for HireSense AI.

Extracts digital text from uploaded PDF resumes using pypdf,
identifies candidate metadata, contact details, and handles edge cases
such as scanned/empty documents.
"""

from __future__ import annotations

import io
import os
import re
from typing import BinaryIO, Dict, Any, List, Optional
import pypdf


EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
LINKEDIN_REGEX = re.compile(
    r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|pub)\/[a-zA-Z0-9_-]+",
    re.IGNORECASE,
)
GITHUB_REGEX = re.compile(
    r"(?:https?:\/\/)?(?:www\.)?github\.com\/[a-zA-Z0-9_-]+",
    re.IGNORECASE,
)


def sanitize_filename_to_name(filename: str) -> str:
    """Fallback candidate name generator from filename.

    Converts 'John_Doe_Resume_2024.pdf' to 'John Doe'.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # Replace separators with spaces first
    spaced = re.sub(r"[-_.]+", " ", base)
    # Remove common boilerplate keywords
    cleaned = re.sub(
        r"(?i)\b(resume|cv|curriculum\s*vitae|profile|final|updated|v\d+|\d{4})\b",
        "",
        spaced,
    ).strip()
    # Normalize extra spaces and capitalize title
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned and len(cleaned) >= 2:
        return cleaned.title()
    return spaced.strip().title()


def extract_candidate_name_heuristic(text: str, fallback_name: str) -> str:
    """Extract candidate name from top lines of the resume text using heuristics."""
    if not text or not text.strip():
        return fallback_name

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_blacklist = {
        "resume",
        "curriculum vitae",
        "cv",
        "summary",
        "professional summary",
        "experience",
        "work experience",
        "education",
        "skills",
        "contact",
        "contact info",
        "personal information",
        "profile",
        "objective",
    }

    # Inspect the first 6 non-empty lines
    for line in lines[:6]:
        clean_line = line.strip()
        lower_line = clean_line.lower()

        # Skip headers or known keywords
        if lower_line in header_blacklist:
            continue
        # Skip if line contains email or phone
        if EMAIL_REGEX.search(clean_line) or PHONE_REGEX.search(clean_line):
            continue
        # Skip if line contains URLs
        if "http" in lower_line or "www." in lower_line or ".com" in lower_line:
            continue
        # Check if line looks like a person's name (2 to 4 words, letters/periods/spaces only)
        words = clean_line.split()
        if 2 <= len(words) <= 4:
            if all(re.match(r"^[A-Za-z\.'-]+$", w) for w in words):
                # Ensure it's not all lowercase unless titled
                return clean_line.title()

    return fallback_name


def extract_text_from_pdf(
    file_source: BinaryIO | bytes,
    filename: str = "Uploaded_Resume.pdf",
) -> Dict[str, Any]:
    """Extracts text, contact details, and candidate information from a PDF file.

    Parameters
    ----------
    file_source : BinaryIO | bytes
        A file-like object or raw bytes of the PDF.
    filename : str
        The original filename.

    Returns
    -------
    dict
        Dictionary containing extracted text, metadata, contact info, and status flags.
    """
    fallback_name = sanitize_filename_to_name(filename)
    result: Dict[str, Any] = {
        "filename": filename,
        "candidate_name": fallback_name,
        "text": "",
        "page_count": 0,
        "char_count": 0,
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "is_valid": False,
        "is_scanned_or_empty": False,
        "error": None,
    }

    try:
        # Normalize into bytes buffer if needed
        if isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
        else:
            # Could be BytesIO or UploadedFile from Streamlit
            if hasattr(file_source, "seek"):
                try:
                    file_source.seek(0)
                except Exception:
                    pass
            content = file_source.read()
            if hasattr(file_source, "seek"):
                try:
                    file_source.seek(0)
                except Exception:
                    pass
            stream = io.BytesIO(content)

        reader = pypdf.PdfReader(stream)

        # Check for password-protected/encrypted PDF
        if getattr(reader, "is_encrypted", False):
            result["error"] = (
                f"The PDF '{filename}' is password-protected or encrypted. "
                "Please upload an unencrypted digital PDF."
            )
            return result

        num_pages = len(reader.pages)
        result["page_count"] = num_pages

        if num_pages == 0:
            result["is_scanned_or_empty"] = True
            result["error"] = "The uploaded PDF document contains 0 pages."
            return result

        extracted_pages: List[str] = []
        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text)
            except Exception as page_err:
                extracted_pages.append("")

        full_text = "\n\n".join(extracted_pages).strip()
        result["text"] = full_text
        result["char_count"] = len(full_text)

        # Detect scanned / image-only PDFs
        if len(full_text.strip()) < 30:
            result["is_scanned_or_empty"] = True
            result["error"] = (
                f"No extractable text found in '{filename}'. "
                "The PDF appears to be a scanned image or non-OCR document. "
                "Please upload a standard text-based digital PDF."
            )
            return result

        result["is_valid"] = True

        # Extract Candidate Name
        extracted_name = extract_candidate_name_heuristic(full_text, fallback_name)
        result["candidate_name"] = extracted_name

        # Extract Contact Details
        emails = EMAIL_REGEX.findall(full_text)
        if emails:
            result["email"] = emails[0]

        phones = PHONE_REGEX.findall(full_text)
        if phones:
            result["phone"] = phones[0]

        linkedin_match = LINKEDIN_REGEX.search(full_text)
        if linkedin_match:
            result["linkedin"] = linkedin_match.group(0)

        github_match = GITHUB_REGEX.search(full_text)
        if github_match:
            result["github"] = github_match.group(0)

        return result

    except pypdf.errors.PdfReadError as pdf_err:
        result["error"] = f"Corrupted or invalid PDF format in '{filename}': {str(pdf_err)}"
        return result
    except Exception as exc:
        result["error"] = f"Failed to parse '{filename}': {str(exc)}"
        return result
