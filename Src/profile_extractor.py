import sys
sys.stdout.reconfigure(encoding="utf-8")

import spacy
import re
import json
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "processed_text"
PROFILE_DIR = BASE_DIR / "data" / "extracted_profiles"

# Load trained spaCy model
nlp = spacy.load("en_core_web_sm")


# Extract candidate name using NER
def extract_name(text):

    doc = nlp(text[:5000])

    for ent in doc.ents:

        if ent.label_ == "PERSON":
            return ent.text

    return None


# Extract education
def extract_education(text):

    education_keywords = [
        "MCA",
        "BCA",
        "MBA",
        "B.Tech",
        "M.Tech",
        "B.Sc",
        "M.Sc",
        "B.Com",
        "M.Com",
        "Bachelor",
        "Master",
        "Ph.D",
        "Diploma"
    ]

    education_found = []

    for line in text.splitlines():

        for keyword in education_keywords:

            if keyword.lower() in line.lower():

                education_found.append(line.strip())
                break

    return list(dict.fromkeys(education_found))


# Extract experience
def extract_experience(text):

    text_lower = text.lower()

    if re.search(r"\bfresher\b|\bno experience\b|\brecent graduate\b", text_lower):
        return "Fresher"

    experience_pattern = (
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?"
        r"(?:\s+of)?\s+experience"
    )

    matches = re.findall(experience_pattern, text_lower)

    if matches:
        return f"{matches[0]} years of experience"

    return "Not specified"


# Process candidate profiles
def process_profiles():

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    text_files = list(INPUT_DIR.glob("*.txt"))

    if not text_files:
        print("No resume text files found!")
        return

    for file_path in text_files:

        print(f"\nProcessing: {file_path.name}")

        text = file_path.read_text(encoding="utf-8")

        # Load previous skill extraction results
        json_path = PROFILE_DIR / f"{file_path.stem}.json"

        if json_path.exists():

            with open(json_path, "r", encoding="utf-8") as file:
                profile = json.load(file)

        else:

            profile = {
                "candidate_id": file_path.stem,
                "email": [],
                "phone": [],
                "technical_skills": []
            }

        # Extract new information
        profile["name"] = extract_name(text)
        profile["education"] = extract_education(text)
        profile["experience"] = extract_experience(text)

        # Save updated profile
        with open(json_path, "w", encoding="utf-8") as file:

            json.dump(
                profile,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("Name:", profile["name"])
        print("Education:", profile["education"])
        print("Experience:", profile["experience"])
        print("Profile updated:", json_path.name)


if __name__ == "__main__":

    process_profiles()
