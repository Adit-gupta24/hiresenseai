"""Script to generate realistic test PDF resumes in data/sample_resumes/

Uses native PDF stream syntax so evaluators, students, or teachers can upload
actual real PDF files to test the system end-to-end.
"""

from __future__ import annotations

import os
try:
    from src.demo_data import (
        SAMPLE_CANDIDATES,
        create_pdf_bytes,
        create_scanned_empty_pdf_bytes,
    )
except (ImportError, ModuleNotFoundError):
    from demo_data import (
        SAMPLE_CANDIDATES,
        create_pdf_bytes,
        create_scanned_empty_pdf_bytes,
    )


def generate_all_sample_pdfs(output_dir: str = "data/sample_resumes") -> list[str]:
    """Generates all sample PDFs and returns their filepaths."""
    os.makedirs(output_dir, exist_ok=True)
    generated: list[str] = []

    for candidate in SAMPLE_CANDIDATES:
        out_path = os.path.join(output_dir, candidate["filename"])
        pdf_bytes = create_pdf_bytes(candidate["text"])
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        generated.append(out_path)

    # Also generate an empty/scanned edge-case PDF to test error handling
    scanned_path = os.path.join(output_dir, "Scanned_Image_Empty_Resume.pdf")
    with open(scanned_path, "wb") as f:
        f.write(create_scanned_empty_pdf_bytes())
    generated.append(scanned_path)

    return generated


if __name__ == "__main__":
    paths = generate_all_sample_pdfs()
    print(f"Successfully generated {len(paths)} sample PDF resumes in data/sample_resumes/")
