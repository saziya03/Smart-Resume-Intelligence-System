
import pymupdf
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output folders
RESUME_DIR = BASE_DIR / "data" / "resumes"
OUTPUT_DIR = BASE_DIR / "data" / "processed_text"


# Function to extract text from one PDF
def extract_resume_text(pdf_path):

    extracted_text = []

    with pymupdf.open(pdf_path) as doc:

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")
            extracted_text.append(text)

            print(f"Page {page_number} extracted successfully")

    return "\n".join(extracted_text)


# Function to process multiple PDFs
def process_all_resumes():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RESUME_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF resumes found!")
        return

    successful = 0
    failed = 0

    print(f"\nTotal resumes found: {len(pdf_files)}")

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        try:

            resume_text = extract_resume_text(pdf_path)

            if not resume_text.strip():
                print("No readable text found in PDF")
                failed += 1
                continue

            output_file = OUTPUT_DIR / f"{pdf_path.stem}.txt"

            output_file.write_text(
                resume_text,
                encoding="utf-8"
            )

            print(f"Saved successfully: {output_file.name}")

            successful += 1

        except Exception as error:

            print(f"Failed: {pdf_path.name}")
            print(f"Reason: {error}")

            failed += 1

    print("\n========== FINAL REPORT ==========")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    process_all_resumes()
