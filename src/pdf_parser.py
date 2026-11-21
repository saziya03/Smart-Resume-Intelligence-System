
"""
Smart Resume Intelligence System
PDF Resume Text Extraction
"""

import logging
from pathlib import Path
import pymupdf as fitz


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "resumes"
OUTPUT_DIR = BASE_DIR / "data" / "processed_text"

RESUME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 3. EXTRACT TEXT FROM ONE PDF
# ==========================================

def extract_resume_text(pdf_path: Path) -> str:

    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Invalid PDF extension.")

    extracted_text = []

    with fitz.open(str(pdf_path)) as doc:

        if len(doc) == 0:
            raise ValueError("PDF contains no pages.")

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            if text:
                extracted_text.append(text)

            logger.info(
                "Page %s extracted from %s",
                page_number,
                pdf_path.name
            )

    return "\n".join(extracted_text).strip()


# ==========================================
# 4. PROCESS ALL PDF RESUMES
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

        logger.warning("No PDF resumes found.")

        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "message": "No PDF resumes found."
        }

    logger.info("Total resumes found: %s", total)

    for pdf_path in pdf_files:

        logger.info("Processing: %s", pdf_path.name)

        try:

            resume_text = extract_resume_text(pdf_path)

            if not resume_text:

                logger.warning(
                    "No readable text found: %s",
                    pdf_path.name
                )

                skipped += 1
                continue

            output_file = OUTPUT_DIR / (
                pdf_path.stem + ".txt"
            )

            output_file.write_text(
                resume_text,
                encoding="utf-8"
            )

            successful += 1

            logger.info(
                "Successfully saved: %s",
                output_file.name
            )

        except Exception:

            logger.exception(
                "Failed to process: %s",
                pdf_path.name
            )

            failed += 1

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
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    result = process_all_resumes()

    print("\n========== PDF PROCESSING REPORT ==========")

    print(f"Total PDFs: {result['total']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Skipped: {result['skipped']}")

    print(result["message"])
