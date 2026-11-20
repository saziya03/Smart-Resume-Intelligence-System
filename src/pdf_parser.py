
"""
Smart Resume Intelligence System
PDF Resume Text Extraction

Supports:
- Multiple PDF resumes
- PDF and pdf extensions
- Text extraction using PyMuPDF
- UTF-8 text file output
- Error handling and logging
"""

import pymupdf as fitz
import logging
from pathlib import Path


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "resumes"
OUTPUT_DIR = BASE_DIR / "data" / "processed_text"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================================
# 2. EXTRACT TEXT FROM ONE PDF
# =====================================================

def extract_resume_text(pdf_path: Path) -> str:

    extracted_text = []

    with fitz.open(str(pdf_path)) as doc:

        if len(doc) == 0:
            raise ValueError("PDF contains no pages.")

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            extracted_text.append(text)

            logger.info(
                "%s: Page %s extracted successfully",
                pdf_path.name,
                page_number
            )

    return "\n".join(extracted_text)


# =====================================================
# 3. PROCESS MULTIPLE PDF RESUMES
# =====================================================

def process_all_resumes():

    # Create required folders
    RESUME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find PDF files (case-insensitive extension)
    pdf_files = sorted(
        file for file in RESUME_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".pdf"
    )

    # Check if PDFs exist
    if not pdf_files:

        logger.warning(
            "No PDF resumes found in %s",
            RESUME_DIR
        )

        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "message": "No PDF resumes found."
        }

    successful = 0
    failed = 0

    logger.info(
        "Total resumes found: %s",
        len(pdf_files)
    )

    # Process every PDF
    for pdf_path in pdf_files:

        logger.info(
            "Processing resume: %s",
            pdf_path.name
        )

        try:

            # Extract text
            resume_text = extract_resume_text(pdf_path)

            # Check extracted content
            if not resume_text.strip():

                logger.warning(
                    "No readable text found in %s",
                    pdf_path.name
                )

                failed += 1
                continue

            # Generate output filename
            output_file = OUTPUT_DIR / (
                pdf_path.stem + ".txt"
            )

            # Save extracted text
            output_file.write_text(
                resume_text,
                encoding="utf-8"
            )

            logger.info(
                "Saved successfully: %s",
                output_file.name
            )

            successful += 1

        except Exception:

            logger.exception(
                "Failed to process PDF: %s",
                pdf_path.name
            )

            failed += 1

    # Final processing report
    report = {
        "total": len(pdf_files),
        "successful": successful,
        "failed": failed,
        "message": "PDF processing completed."
    }

    logger.info(
        "FINAL REPORT: %s",
        report
    )

    return report


# =====================================================
# 4. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    result = process_all_resumes()

    print("\n========== PDF PROCESSING REPORT ==========")

    print(f"Total PDFs: {result['total']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")

    print(result["message"])
