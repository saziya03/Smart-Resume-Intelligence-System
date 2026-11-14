# Smart Resume Intelligence System

An NLP and Retrieval-Augmented Generation (RAG) based application that analyzes resumes, extracts candidate information, matches resumes with job descriptions, ranks candidates, and answers recruiter questions using AI.

## Project Overview

Smart Resume Intelligence System helps recruiters reduce manual resume screening by automating candidate analysis and job matching.

The system supports multiple PDF resumes and provides candidate ranking based on skills and semantic similarity.

## Key Features

- Multiple PDF Resume Upload
- Resume Text Extraction
- NLP-based Skill Extraction
- Candidate Profile Extraction
- Job Description Analysis
- Semantic Resume Matching
- Candidate Ranking Dashboard
- AI Recruiter Assistant using RAG
- Vector Search using ChromaDB

## Technologies Used

**Programming:** Python

**NLP:** spaCy, Regex, Sentence Transformers

**Machine Learning:** Semantic Similarity, Embeddings

**Generative AI:** RAG, LLM Integration

**Vector Database:** ChromaDB

**PDF Processing:** PyMuPDF

**Frontend:** Streamlit

**Data Processing:** Pandas, NumPy

## Project Architecture

1. Upload multiple PDF resumes.
2. Extract text from resumes.
3. Clean and preprocess extracted text.
4. Extract skills and candidate profiles.
5. Parse the job description.
6. Generate semantic embeddings.
7. Store and retrieve resume information using ChromaDB.
8. Calculate candidate matching scores.
9. Rank candidates.
10. Generate answers using the RAG-based recruiter assistant.

## Project Structure

```text
Smart-Resume-Intelligence-System/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── pdf_parser.py
│   ├── skill_extractor.py
│   ├── profile_extractor.py
│   ├── job_description_parser.py
│   ├── matching_engine.py
│   └── rag_engine.py
│
├── data/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

*Update this structure if your actual folder names differ.*

## Installation

Clone the repository:

```bash
git clone https://github.com/saziya03/Smart-Resume-Intelligence-System.git

cd Smart-Resume-Intelligence-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app/streamlit_app.py
```

## Future Improvements

- Advanced Named Entity Recognition
- Improved semantic matching
- RAG retrieval evaluation
- FastAPI backend integration
- Cloud deployment
- Better candidate ranking evaluation

## Author

**Saziya Alvi**

GitHub: https://github.com/saziya03

LinkedIn: https://linkedin.com/in/saziya-alvi
