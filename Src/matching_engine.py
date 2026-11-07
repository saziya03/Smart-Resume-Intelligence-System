
import json
import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "processed_text"
PROFILE_DIR = BASE_DIR / "data" / "extracted_profiles"
JD_DIR = BASE_DIR / "data" / "processed_jds"

OUTPUT_DIR = BASE_DIR / "data" / "matching_results"

# Load the pre-trained NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_skill_match(resume_skills, required_skills):

    resume_skills = {skill.lower() for skill in resume_skills}
    required_skills = {skill.lower() for skill in required_skills}

    if not required_skills:
        return 0, []

    matched = resume_skills.intersection(required_skills)
    missing = required_skills - resume_skills

    score = (len(matched) / len(required_skills)) * 100

    return score, sorted(missing)


def calculate_semantic_similarity(resume_text, jd_text):

    embeddings = model.encode([resume_text, jd_text])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return max(0, float(similarity)) * 100


def match_resumes():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    jd_files = list(JD_DIR.glob("*.json"))

    if not jd_files:
        print("No processed job descriptions found!")
        return

    resume_files = list(RESUME_DIR.glob("*.txt"))

    if not resume_files:
        print("No processed resume text files found!")
        return

    # Use the first JD for this initial version
    jd_data = load_json(jd_files[0])
    jd_text = jd_data["job_description"]
    required_skills = jd_data["required_skills"]

    results = []

    for resume_file in resume_files:

        print(f"\nMatching: {resume_file.name}")

        resume_text = resume_file.read_text(encoding="utf-8")

        profile_file = PROFILE_DIR / f"{resume_file.stem}.json"

        if profile_file.exists():
            profile = load_json(profile_file)
        else:
            profile = {}

        resume_skills = profile.get("technical_skills", [])

        semantic_score = calculate_semantic_similarity(
            resume_text, jd_text
        )

        skill_score, missing_skills = calculate_skill_match(
            resume_skills, required_skills
        )

        # Initial weighted matching score
        final_score = (
            0.70 * semantic_score +
            0.30 * skill_score
        )

        results.append({
            "candidate": resume_file.stem,
            "semantic_score": round(semantic_score, 2),
            "skill_score": round(skill_score, 2),
            "final_score": round(final_score, 2),
            "missing_skills": ", ".join(missing_skills)
        })

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    output_file = OUTPUT_DIR / "candidate_ranking.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "candidate",
                "semantic_score",
                "skill_score",
                "final_score",
                "missing_skills"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== CANDIDATE RANKING ==========")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['candidate']} "
            f"| Match: {result['final_score']}%"
        )
        print("Missing skills:", result["missing_skills"])

    print("\nResults saved:", output_file)


if __name__ == "__main__":
    match_resumes()
