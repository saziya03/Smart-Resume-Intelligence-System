
import json
import csv
import re
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "processed_text"
PROFILE_DIR = BASE_DIR / "data" / "extracted_profiles"
JD_DIR = BASE_DIR / "data" / "processed_jds"

OUTPUT_DIR = BASE_DIR / "data" / "matching_results"

OUTPUT_FILE = OUTPUT_DIR / "candidate_ranking.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# 2. LOAD NLP MODEL
# =====================================================

model = SentenceTransformer(MODEL_NAME)


# =====================================================
# 3. SKILL NORMALIZATION
# =====================================================

ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "llm": "large language models",
    "llms": "large language models",
    "sklearn": "scikit-learn",
    "powerbi": "power bi",
    "ms excel": "excel",
    "nodejs": "node.js",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
}


def normalize_skill(skill):

    skill = str(skill).strip().lower()

    skill = re.sub(r"\s+", " ", skill)

    return ALIASES.get(skill, skill)


def normalize_skill_list(skills):

    if isinstance(skills, str):
        skills = [skills]

    if not isinstance(skills, list):
        return set()

    return {
        normalize_skill(skill)
        for skill in skills
        if skill and str(skill).strip()
    }


# =====================================================
# 4. JSON LOADER
# =====================================================

def load_json(file_path):

    try:

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:

        logger.warning(
            "Could not read %s: %s",
            file_path.name,
            error
        )

        return {}


# =====================================================
# 5. RESUME PROFILE DATA
# =====================================================

def get_resume_skills(profile, resume_text, jd_data):

    skills = set()

    # Skills extracted by profile extractor
    possible_keys = [
        "technical_skills",
        "soft_skills",
        "skills",
        "all_skills",
        "programming_languages",
        "tools",
        "frameworks",
        "libraries",
        "technologies"
    ]

    for key in possible_keys:

        value = profile.get(key, [])

        if isinstance(value, list):
            skills.update(normalize_skill_list(value))

        elif isinstance(value, str):
            skills.update(normalize_skill_list([value]))

    # Also detect JD-specific skills directly in resume text.
    # This helps when profile extractor uses a different schema.

    jd_skills = (
        jd_data.get("all_skills", [])
        + jd_data.get("required_skills", [])
        + jd_data.get("preferred_skills", [])
    )

    resume_lower = resume_text.lower()

    for skill in jd_skills:

        skill_lower = str(skill).lower().strip()

        if not skill_lower:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(skill_lower)
            + r"(?!\w)"
        )

        if re.search(pattern, resume_lower):

            skills.add(normalize_skill(skill))

    return skills


# =====================================================
# 6. SKILL MATCHING
# =====================================================

def calculate_skill_match(
    resume_skills,
    required_skills,
    preferred_skills=None
):

    preferred_skills = preferred_skills or []

    resume_skills = normalize_skill_list(
        list(resume_skills)
    )

    required = normalize_skill_list(required_skills)

    preferred = normalize_skill_list(preferred_skills)

    matched_required = resume_skills.intersection(required)

    missing_required = required - resume_skills

    matched_preferred = resume_skills.intersection(preferred)

    missing_preferred = preferred - resume_skills

    # Required skills have higher importance.
    if required:

        required_score = (
            len(matched_required) / len(required)
        ) * 100

    else:
        required_score = 0

    if preferred:

        preferred_score = (
            len(matched_preferred) / len(preferred)
        ) * 100

    else:
        preferred_score = None

    # Preferred skills cannot compensate fully
    # for missing mandatory skills.

    if preferred_score is not None:

        skill_score = (
            0.80 * required_score
            + 0.20 * preferred_score
        )

    else:
        skill_score = required_score

    return {
        "skill_score": round(skill_score, 2),
        "required_skill_score": round(required_score, 2),
        "matched_skills": sorted(matched_required),
        "missing_skills": sorted(missing_required),
        "matched_preferred_skills": sorted(matched_preferred),
        "missing_preferred_skills": sorted(missing_preferred)
    }


# =====================================================
# 7. SEMANTIC SIMILARITY
# =====================================================

def calculate_semantic_similarity(resume_text, jd_text):

    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    embeddings = model.encode(
        [resume_text, jd_text],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    # Cosine similarity can be negative.
    # Clamp it to a 0-100 display range.

    score = max(0.0, float(similarity)) * 100

    return round(score, 2)


# =====================================================
# 8. EXPERIENCE MATCHING
# =====================================================

def extract_years(text):

    if not text:
        return None

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            str(text),
            re.IGNORECASE
        )

        if matches:

            try:
                return max(float(item) for item in matches)
            except ValueError:
                pass

    return None


def get_candidate_experience(profile, resume_text):

    keys = [
        "total_experience",
        "experience_years",
        "years_of_experience",
        "experience"
    ]

    for key in keys:

        value = profile.get(key)

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):

            years = extract_years(value)

            if years is not None:
                return years

    return extract_years(resume_text)


def calculate_experience_score(
    candidate_years,
    jd_experience
):

    required_years = extract_years(jd_experience)

    if required_years is None or candidate_years is None:
        return None

    if required_years <= 0:
        return None

    score = min(
        candidate_years / required_years,
        1.0
    ) * 100

    return round(score, 2)


# =====================================================
# 9. HIRING RECOMMENDATION
# =====================================================

def get_recommendation(score):

    if score >= 85:
        return "Strong Match"

    elif score >= 70:
        return "Good Match"

    elif score >= 50:
        return "Partial Match"

    else:
        return "Low Match"


# =====================================================
# 10. FINAL MATCH SCORE
# =====================================================

def calculate_final_score(
    semantic_score,
    skill_score,
    experience_score
):

    # Base weights
    components = [
        (semantic_score, 0.40),
        (skill_score, 0.60)
    ]

    # Experience is included only when
    # both resume and JD contain usable data.
    if experience_score is not None:

        components = [
            (semantic_score, 0.35),
            (skill_score, 0.55),
            (experience_score, 0.10)
        ]

    total_weight = sum(
        weight for _, weight in components
    )

    score = sum(
        value * weight
        for value, weight in components
    ) / total_weight

    return round(score, 2)


# =====================================================
# 11. LOAD ALL RESUMES
# =====================================================

def load_resumes():

    resumes = []

    if not RESUME_DIR.exists():
        logger.warning("Resume directory does not exist.")
        return resumes

    resume_files = sorted(RESUME_DIR.glob("*.txt"))

    for resume_file in resume_files:

        try:

            resume_text = resume_file.read_text(
                encoding="utf-8"
            )

            profile_file = (
                PROFILE_DIR / f"{resume_file.stem}.json"
            )

            profile = load_json(profile_file)

            resumes.append({
                "candidate": resume_file.stem,
                "resume_text": resume_text,
                "profile": profile
            })

        except OSError as error:

            logger.warning(
                "Resume skipped: %s",
                error
            )

    return resumes


# =====================================================
# 12. MATCH ONE RESUME AGAINST ONE JD
# =====================================================

def match_candidate_to_job(candidate, jd_data):

    resume_text = candidate["resume_text"]

    profile = candidate["profile"]

    jd_text = jd_data.get("job_description", "")

    required_skills = jd_data.get("required_skills", [])

    preferred_skills = jd_data.get("preferred_skills", [])

    candidate_skills = get_resume_skills(
        profile,
        resume_text,
        jd_data
    )

    skill_results = calculate_skill_match(
        candidate_skills,
        required_skills,
        preferred_skills
    )

    semantic_score = calculate_semantic_similarity(
        resume_text,
        jd_text
    )

    candidate_years = get_candidate_experience(
        profile,
        resume_text
    )

    jd_experience = " ".join(
        jd_data.get("experience_requirements", [])
    )

    experience_score = calculate_experience_score(
        candidate_years,
        jd_experience
    )

    final_score = calculate_final_score(
        semantic_score,
        skill_results["skill_score"],
        experience_score
    )

    return {
        "candidate": candidate["candidate"],
        "job_id": jd_data.get("job_id", ""),
        "job_title": jd_data.get("job_title", "Not specified"),

        "semantic_score": semantic_score,

        "skill_score": skill_results["skill_score"],

        "required_skill_score": skill_results[
            "required_skill_score"
        ],

        "experience_score": (
            experience_score
            if experience_score is not None
            else "Not Available"
        ),

        "candidate_experience_years": (
            candidate_years
            if candidate_years is not None
            else "Not Available"
        ),

        "final_score": final_score,

        "matched_skills": ", ".join(
            skill_results["matched_skills"]
        ),

        "missing_skills": ", ".join(
            skill_results["missing_skills"]
        ),

        "matched_preferred_skills": ", ".join(
            skill_results["matched_preferred_skills"]
        ),

        "missing_preferred_skills": ", ".join(
            skill_results["missing_preferred_skills"]
        ),

        "recommendation": get_recommendation(final_score)
    }


# =====================================================
# 13. MATCH ALL RESUMES AGAINST ALL JOBS
# =====================================================

def match_resumes():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not JD_DIR.exists():
        logger.error("Processed JD directory not found.")
        return

    jd_files = sorted(JD_DIR.glob("*.json"))

    if not jd_files:
        logger.error("No processed job descriptions found.")
        return

    candidates = load_resumes()

    if not candidates:
        logger.error("No processed resumes found.")
        return

    all_results = []

    print("\n========== UNIVERSAL RESUME MATCHING ==========")

    print("Total Resumes:", len(candidates))
    print("Total Job Descriptions:", len(jd_files))

    # Process every job description
    for jd_file in jd_files:

        jd_data = load_json(jd_file)

        if not jd_data:
            continue

        print(
            f"\nProcessing Job: "
            f"{jd_data.get('job_title', jd_file.stem)}"
        )

        # Match every candidate against this JD
        for candidate in candidates:

            print(
                "Matching:",
                candidate["candidate"]
            )

            result = match_candidate_to_job(
                candidate,
                jd_data
            )

            all_results.append(result)

    if not all_results:
        logger.error("No matching results generated.")
        return

    # Save all candidate-job combinations
    all_results.sort(
        key=lambda item: (
            item["job_id"],
            -item["final_score"]
        )
    )

    fieldnames = [
        "candidate",
        "job_id",
        "job_title",
        "semantic_score",
        "skill_score",
        "required_skill_score",
        "experience_score",
        "candidate_experience_years",
        "final_score",
        "matched_skills",
        "missing_skills",
        "matched_preferred_skills",
        "missing_preferred_skills",
        "recommendation"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Print ranking separately for each JD
    for jd_file in jd_files:

        jd_data = load_json(jd_file)

        if not jd_data:
            continue

        job_id = jd_data.get("job_id", jd_file.stem)

        job_results = [
            item for item in all_results
            if item["job_id"] == job_id
        ]

        job_results.sort(
            key=lambda item: item["final_score"],
            reverse=True
        )

        print(
            f"\n========== RANKING: "
            f"{jd_data.get('job_title', job_id)} =========="
        )

        for rank, result in enumerate(
            job_results,
            start=1
        ):

            print(
                f"{rank}. {result['candidate']} "
                f"| Match: {result['final_score']}% "
                f"| {result['recommendation']}"
            )

            print(
                "Matched:",
                result["matched_skills"]
            )

            print(
                "Missing:",
                result["missing_skills"]
            )

    print("\n========================================")
    print("Total Matching Results:", len(all_results))
    print("Results Saved:", OUTPUT_FILE)
    print("========================================")


# =====================================================
# 14. RUN
# =====================================================

if __name__ == "__main__":
    match_resumes()
