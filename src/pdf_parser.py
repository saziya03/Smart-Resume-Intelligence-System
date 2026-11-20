
"""
Smart Resume Intelligence System
PDF Resume Text Extraction

Features:
- Multiple PDF resume processing
- PyMuPDF text extraction
- Case-insensitive PDF extensions
- UTF-8 output
- PDF validation
- Error handling and logging
"""

import logging
from pathlib import Path

# ==========================================
# 1. IMPORT PDF LIBRARY
# ==========================================

try:
    import pymupdf as fitz

except ImportError:
    try:
        import fitz

    except ImportError:
        raise ImportError(
            "PyMuPDF is not installed. "
            "Add PyMuPDF to requirements.txt"
        )


# ==========================================
# 2. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "resumes"
OUTPUT_DIR = BASE_DIR / "data" / "processed_text"

RESUME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 4. EXTRACT TEXT FROM SINGLE PDF
# ==========================================

def extract_resume_text(pdf_path: Path) -> str:

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Resume not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Invalid PDF file extension.")

    extracted_text = []

    with fitz.open(str(pdf_path)) as doc:

        if len(doc) == 0:
            raise ValueError("PDF contains no pages.")

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            if text:
                extracted_text.append(text)

            logger.info(
                "Extracted page %s from %s",
                page_number,
                pdf_path.name
            )

    return "\n".join(extracted_text).strip()


# ==========================================
# 5. PROCESS ALL PDF RESUMES
# ==========================================

def process_all_resumes():

    pdf_files = sorted(
        file for file in RESUME_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".pdf"
    )

    total = len(pdf_files)
    successful = 0
    failed = 0
    skipped = 0

    if total == 0:

        logger.warning(
            "No PDF resumes found in %s",
            RESUME_DIR
        )

        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "message": "No PDF resumes found."
        }

    logger.info("Total PDF resumes found: %s", total)

    # Process each resume

    for pdf_path in pdf_files:

        logger.info(
            "Processing resume: %s",
            pdf_path.name
        )

        try:

            # Extract PDF text
            resume_text = extract_resume_text(pdf_path)

            # Check extracted content
            if not resume_text:

                logger.warning(
                    "No readable text found in %s. "
                    "PDF may be scanned.",
                    pdf_path.name
                )

                skipped += 1
                continue

            # Save extracted text
            output_file = OUTPUT_DIR / (
                pdf_path.stem + ".txt"
            )

            output_file.write_text(
                resume_text,
                encoding="utf-8"
            )

            logger.info(
                "Successfully saved: %s",
                output_file.name
            )

            successful += 1

        except Exception as error:

            logger.exception(
                "Error processing %s: %s",
                pdf_path.name,
                error
            )

            failed += 1

    # ======================================
    # 6. FINAL REPORT
    # ======================================

    report = {
        "total": total,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "message": "PDF processing completed."
    }

    logger.info("FINAL REPORT: %s", report)

    return report


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    result = process_all_resumes()

    print("\n========== PDF PROCESSING REPORT ==========")

    print(f"Total PDFs: {result['total']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Skipped: {result['skipped']}")

    print(result["message"])
