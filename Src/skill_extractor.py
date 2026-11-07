
import spacy
import re
import json
from pathlib import Path
from spacy.matcher import PhraseMatcher


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "processed_text"
OUTPUT_DIR = BASE_DIR / "data" / "extracted_profiles"


# Initialize spaCy
nlp = spacy.blank("en")

# Create PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")


# Technical skills dictionary
SKILLS = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "MySQL",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "NLP",
    "Artificial Intelligence",
    "AI",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "BERT",
    "Hugging Face",
    "LangChain",
    "RAG",
    "Generative AI",
    "Power BI",
    "Tableau",
    "Excel",
    "Statistics",
    "Data Analysis",
    "Data Visualization",
    "FastAPI",
    "Flask",
    "Streamlit",
    "Git",
    "GitHub",
    "Docker",
    "AWS"
]


# Convert skills into spaCy patterns
patterns = [nlp.make_doc(skill) for skill in SKILLS]

matcher.add("TECHNICAL_SKILLS", patterns)


# Extract email
def extract_email(text):

    emails = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    return list(set(emails))


# Extract phone number
def extract_phone(text):

    phones = re.findall(
        r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)",
        text
    )

    return list(set(phones))


# Extract technical skills
def extract_skills(text):

    doc = nlp(text)

    matches = matcher(doc)

    extracted_skills = set()

    for match_id, start, end in matches:

        span = doc[start:end]

        extracted_skills.add(span.text)

    return sorted(extracted_skills, key=str.lower)


# Process all extracted resumes
def process_resumes():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text_files = list(INPUT_DIR.glob("*.txt"))

    if not text_files:
        print("No extracted resume text files found!")
        return

    print(f"Total resumes found: {len(text_files)}")

    for file_path in text_files:

        print(f"\nProcessing: {file_path.name}")

        text = file_path.read_text(encoding="utf-8")

        profile = {
            "candidate_id": file_path.stem,
            "email": extract_email(text),
            "phone": extract_phone(text),
            "technical_skills": extract_skills(text)
        }

        output_file = OUTPUT_DIR / f"{file_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as file:

            json.dump(
                profile,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("Extracted Skills:", profile["technical_skills"])
        print("Profile saved:", output_file.name)


if __name__ == "__main__":

    process_resumes()
