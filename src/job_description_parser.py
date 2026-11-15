
import re
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

JD_DIR = BASE_DIR / "data" / "job_descriptions"
OUTPUT_DIR = BASE_DIR / "data" / "processed_jds"

# Common skill aliases
SKILL_ALIASES = {
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "gen ai": "Generative AI",
    "genai": "Generative AI",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "sklearn": "Scikit-learn",
    "powerbi": "Power BI",
    "ms excel": "Excel",
    "rest api": "REST API",
    "restful api": "REST API",
    "node.js": "Node.js",
    "react.js": "React",
    "amazon web services": "AWS",
    "google cloud platform": "GCP"
}

# Broad vocabulary: an initial recognition layer,
# not a restriction on which job titles are supported.
KNOWN_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle",
    "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "Natural Language Processing", "Generative AI",
    "Large Language Models", "Computer Vision",
    "Pandas", "NumPy", "Scikit-learn",
    "TensorFlow", "PyTorch", "Keras",
    "BERT", "Hugging Face", "Transformers",
    "RAG", "LangChain", "LangGraph", "LlamaIndex",
    "ChromaDB", "FAISS", "Vector Database",
    "Prompt Engineering", "Fine-tuning", "Embeddings",
    "Text Preprocessing", "Tokenization", "TF-IDF",
    "Power BI", "Tableau", "Excel",
    "Statistics", "Data Analysis", "Data Visualization",
    "ETL", "Data Warehousing", "Apache Spark",
    "Hadoop", "Airflow", "Kafka",
    "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Git", "GitHub",
    "FastAPI", "Flask", "Django", "REST API",
    "React", "Node.js", "HTML", "CSS",
    "Linux", "Networking", "Cybersecurity",
    "Selenium", "Pytest", "Jenkins",
    "Agile", "Scrum", "JIRA",
    "Communication", "Leadership",
    "Problem Solving", "Teamwork",
    "Project Management", "Critical Thinking",
    "Sales", "Marketing", "Recruitment",
    "Financial Analysis", "Accounting",
    "Customer Service", "Business Analysis",
    "SEO", "Content Writing"
]


def normalize_text(text):
    text = text.replace("\u2022", " ")

    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line)
        lines.append(line.strip())

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def canonical_skill(skill):
    """Convert skill aliases into standard names."""

    return SKILL_ALIASES.get(
        skill.lower().strip(),
        skill.strip()
    )


def extract_skills(text):
    """
    Extract recognized skills from JD.
    Supports aliases and word-boundary matching.
    """

    found = {}

    for skill in KNOWN_SKILLS:

        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text, re.IGNORECASE):
            standard_name = canonical_skill(skill)
            found[standard_name.lower()] = standard_name

    # Alias matching
    for alias, standard in SKILL_ALIASES.items():

        pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"

        if re.search(pattern, text, re.IGNORECASE):
            found[standard.lower()] = standard

    return sorted(found.values())


def extract_job_title(text):
    """Try to identify the job title from common JD headings."""

    patterns = [
        r"(?im)^\s*(?:job title|position|role|designation)\s*[:\-]\s*(.+)$",
        r"(?im)^\s*(?:we are hiring|hiring for)\s*[:\-]?\s*(.+)$"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group(1).strip()[:150]

    return "Not specified"


def extract_experience(text):
    """Extract experience requirements from job description."""

    patterns = [
        r"(\d+\+?\s*(?:-|to)\s*\d+\s*years?)",
        r"(\d+\+?\s*years?\s+(?:of\s+)?experience)",
        r"(fresher|entry[- ]level|internship)"
    ]

    results = []

    for pattern in patterns:

        matches = re.findall(pattern, text, re.IGNORECASE)

        results.extend(matches)

    return list(dict.fromkeys(
        str(item).strip() for item in results
    ))


def extract_sections(text):
    """Extract common JD sections."""

    section_patterns = {
        "responsibilities": [
            "responsibilities",
            "what you'll do",
            "key duties"
        ],
        "qualifications": [
            "qualifications",
            "requirements",
            "what we're looking for"
        ],
        "preferred_qualifications": [
            "preferred qualifications",
            "good to have",
            "nice to have",
            "preferred skills"
        ]
    }

    lines = text.splitlines()
    sections = {
        "responsibilities": "",
        "qualifications": "",
        "preferred_qualifications": ""
    }

    current_section = None

    for line in lines:

        cleaned = line.strip()
        lower = cleaned.lower().strip(" :")

        matched_section = None

        for section, headings in section_patterns.items():

            if any(
                lower == heading or lower.startswith(heading + ":")
                for heading in headings
            ):
                matched_section = section
                break

        if matched_section:
            current_section = matched_section
            continue

        if current_section and cleaned:
            sections[current_section] += cleaned + "\n"

    return {
        key: value.strip()
        for key, value in sections.items()
    }


def parse_job_description(text, job_id="manual_jd"):
    """Main universal JD parsing function."""

    text = normalize_text(text)

    skills = extract_skills(text)

    sections = extract_sections(text)

    preferred_text = sections["preferred_qualifications"]

    preferred_skills = extract_skills(preferred_text)

    required_skills = [
        skill for skill in skills
        if skill not in preferred_skills
    ]

    return {
        "job_id": job_id,
        "job_title": extract_job_title(text),
        "job_description": text,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "all_skills": skills,
        "experience_requirements": extract_experience(text),
        "responsibilities": sections["responsibilities"],
        "qualifications": sections["qualifications"],
        "preferred_qualifications": preferred_text,
        "processed_at": datetime.now().isoformat()
    }


def process_job_descriptions():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not JD_DIR.exists():
        JD_DIR.mkdir(parents=True, exist_ok=True)
        print("Created job_descriptions directory.")
        return

    jd_files = list(JD_DIR.glob("*.txt"))

    if not jd_files:
        print("No job descriptions found!")
        return

    for jd_file in jd_files:

        print(f"\nProcessing: {jd_file.name}")

        text = jd_file.read_text(encoding="utf-8")

        profile = parse_job_description(
            text,
            job_id=jd_file.stem
        )

        output_file = OUTPUT_DIR / f"{jd_file.stem}.json"

        with open(output_file, "w", encoding="utf-8") as file:

            json.dump(
                profile,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("Job Title:", profile["job_title"])
        print("Required Skills:", profile["required_skills"])
        print("Preferred Skills:", profile["preferred_skills"])
        print("Saved:", output_file.name)


if __name__ == "__main__":
    process_job_descriptions()
