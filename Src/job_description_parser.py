

import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

JD_DIR = BASE_DIR / "data" / "job_descriptions"
OUTPUT_DIR = BASE_DIR / "data" / "processed_jds"

SKILLS = [
    "Python", "Java", "SQL", "MySQL",
    "Machine Learning", "Deep Learning",
    "Natural Language Processing", "NLP",
    "Pandas", "NumPy", "Scikit-learn",
    "TensorFlow", "PyTorch", "BERT",
    "Hugging Face", "Transformers",
    "Text Preprocessing", "Tokenization",
    "Stemming", "Lemmatization",
    "TF-IDF", "Word Embeddings",
    "Git", "Docker", "FastAPI"
]

def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)

    return found_skills


def process_job_descriptions():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jd_files = list(JD_DIR.glob("*.txt"))

    if not jd_files:
        print("No job descriptions found!")
        return

    for jd_file in jd_files:

        print(f"\nProcessing: {jd_file.name}")

        text = jd_file.read_text(encoding="utf-8")

        profile = {
            "job_id": jd_file.stem,
            "job_description": text,
            "required_skills": extract_skills(text)
        }

        output_file = OUTPUT_DIR / f"{jd_file.stem}.json"

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(profile, file, indent=4)

        print("Required Skills:", profile["required_skills"])
        print("Saved:", output_file.name)


if __name__ == "__main__":
    process_job_descriptions()
