"""
Smart Resume Intelligence System - Streamlit application

Expected location:
    Smart-Resume-Intelligence-System/app/streamlit_app.py

Run from the project root:
    streamlit run app/streamlit_app.py

Expected project folders:
    src/, data/resumes/, data/processed_text/,
    data/job_descriptions/, data/matching_results/, models/
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

SRC_DIR = PROJECT_ROOT / "src"
RESUME_DIR = PROJECT_ROOT / "data" / "resumes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_text"
JOB_DESCRIPTION_DIR = PROJECT_ROOT / "data" / "job_descriptions"
RESULT_DIR = PROJECT_ROOT / "data" / "matching_results"

RANKING_FILE = RESULT_DIR / "candidate_ranking.csv"
JOB_DESCRIPTION_FILE = JOB_DESCRIPTION_DIR / "current_job.txt"

for folder in (
    RESUME_DIR,
    PROCESSED_DIR,
    JOB_DESCRIPTION_DIR,
    RESULT_DIR,
    PROJECT_ROOT / "models",
):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Resume Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 Smart Resume Intelligence System")
st.caption(
    "Upload resumes, compare candidates with a job description, "
    "and ask questions using the AI Recruiter Assistant."
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def show_project_structure_warning() -> None:
    """Show a clear warning if the expected source folder is missing."""
    if not SRC_DIR.exists():
        st.error(f"Source folder not found: {SRC_DIR}")
        st.info(
            "Your source folder must be named exactly 'src' "
            "and should be inside the main project folder."
        )
        st.stop()

    if not (SRC_DIR / "__init__.py").exists():
        st.warning(
            "src/__init__.py was not found. Create an empty __init__.py "
            "file inside the src folder so Python can import src modules."
        )


def save_uploaded_resumes(uploaded_files: list[Any]) -> int:
    """Save uploaded PDF files in data/resumes."""
    saved_count = 0

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            st.warning(f"Skipped {uploaded_file.name}: only PDF files are supported.")
            continue

        # Keep the original filename but remove directory components.
        safe_name = Path(uploaded_file.name).name
        destination = RESUME_DIR / safe_name
        destination.write_bytes(uploaded_file.getvalue())
        saved_count += 1

    return saved_count


def run_python_script(script_name: str) -> tuple[bool, str]:
    """
    Run one Python pipeline script from the project root.

    The scripts are expected to support command-line execution, for example:
        python src/pdf_parser.py
    """
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )

        output = (result.stdout or "").strip()
        errors = (result.stderr or "").strip()

        if result.returncode != 0:
            message = (
                f"{script_name} failed with exit code {result.returncode}.\n\n"
                f"STDOUT:\n{output or '(No standard output)'}\n\n"
                f"STDERR:\n{errors or '(No error output)'}"
            )
            return False, message

        combined = output
        if errors:
            combined += ("\n\nWarnings / logs:\n" if combined else "") + errors

        return True, combined or f"{script_name} completed successfully."

    except subprocess.TimeoutExpired:
        return False, f"{script_name} timed out after 10 minutes."
    except Exception as exc:
        return False, f"Could not run {script_name}: {exc}"


def run_resume_pipeline() -> tuple[bool, list[tuple[str, str]]]:
    """
    Execute the project scripts in order.

    If your scripts use a different execution method or filenames,
    update this list to match your actual src folder.
    """
    pipeline_scripts = [
        "pdf_parser.py",
        "skill_extractor.py",
        "profile_extractor.py",
        "job_description_parser.py",
        "matching_engine.py",
    ]

    logs: list[tuple[str, str]] = []

    for script_name in pipeline_scripts:
        success, output = run_python_script(script_name)
        logs.append((script_name, output))

        if not success:
            return False, logs

    # Build/update the resume vector database after text extraction.
    try:
        from src.rag_engine import store_resumes

        store_resumes()
        logs.append(("RAG database", "Resume knowledge database updated successfully."))
    except Exception as exc:
        logs.append(("RAG database", f"Could not update RAG database: {exc}"))
        return False, logs

    return True, logs


def load_ranking_data() -> pd.DataFrame:
    """Read candidate ranking CSV safely."""
    if not RANKING_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(RANKING_FILE)
        if df.empty:
            return pd.DataFrame()
        return df
    except pd.errors.EmptyDataError:
        st.warning("The candidate ranking CSV is empty.")
        return pd.DataFrame()
    except Exception as exc:
        st.error(f"Could not read candidate ranking CSV: {exc}")
        return pd.DataFrame()


def find_score_column(df: pd.DataFrame) -> str | None:
    """Find a likely candidate matching score column."""
    possible_columns = [
        "final_score",
        "match_score",
        "similarity_score",
        "score",
    ]

    lower_to_original = {str(col).lower(): col for col in df.columns}
    for column in possible_columns:
        if column in lower_to_original:
            return lower_to_original[column]

    return None


def get_rag_answer(question: str, chat_history: list[dict[str, str]]) -> str:
    """Call the RAG chatbot using one consistent function signature."""
    from src.rag_engine import generate_answer

    answer = generate_answer(
        question=question,
        chat_history=chat_history,
        top_k=5,
    )

    if answer is None:
        return "The AI assistant did not return an answer."

    return str(answer)


# ============================================================
# 4. SIDEBAR - PROJECT STATUS
# ============================================================

with st.sidebar:
    st.header("⚙️ Project Status")
    st.write("**Project root:**")
    st.code(str(PROJECT_ROOT), language="text")

    st.write("**Source folder:**")
    if SRC_DIR.exists():
        st.success("src folder found")
    else:
        st.error("src folder missing")

    pdf_count = len(list(RESUME_DIR.glob("*.pdf")))
    text_count = len(list(PROCESSED_DIR.glob("*.txt")))

    st.metric("PDF resumes", pdf_count)
    st.metric("Processed text files", text_count)

    if st.button("Refresh application", width='stretch' ):
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear All / New Session", width='stretch' ):
        import shutil, glob, os
        for f in glob.glob("data/resumes/*"):
            os.remove(f)
        for f in glob.glob("data/processed_text/*"):
            os.remove(f)
        for f in glob.glob("data/extracted_profiles/*"):
            os.remove(f)
        for f in glob.glob("data/matching_results/*"):
            os.remove(f)
        if os.path.exists("models/resume_vector_db"):
            shutil.rmtree("models/resume_vector_db")
        st.rerun()

# ============================================================
# 5. RESUME UPLOAD AND JOB DESCRIPTION
# ============================================================

st.header("1. Upload Resumes")

uploaded_files = st.file_uploader(
    "Choose one or more candidate resume PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload PDF resumes. Files are saved in data/resumes/.",
)

if uploaded_files:
    st.write(f"Selected files: **{len(uploaded_files)}**")

    if st.button("Save Uploaded Resumes", type="primary"):
        count = save_uploaded_resumes(uploaded_files)
        if count:
            st.success(f"{count} resume(s) saved successfully.")
        else:
            st.warning("No PDF resumes were saved.")

st.divider()

st.header("2. Enter Job Description")

job_title = st.text_input(
    "Job title",
    placeholder="Example: NLP Engineer",
)

job_description = st.text_area(
    "Job description",
    height=220,
    placeholder=(
        "Paste the complete job description here, including required skills, "
        "experience, responsibilities, and qualifications."
    ),
)

if st.button("Save Job Description"):
    if not job_title.strip() or not job_description.strip():
        st.warning("Please enter both the job title and job description.")
    else:
        JOB_DESCRIPTION_DIR.mkdir(parents=True, exist_ok=True)
        JOB_DESCRIPTION_FILE.write_text(
            f"Job Title: {job_title.strip()}\n\n"
            f"Job Description:\n{job_description.strip()}",
            encoding="utf-8",
        )
        st.success(f"Job description saved: {JOB_DESCRIPTION_FILE.name}")


# ============================================================
# 6. RUN RESUME PROCESSING PIPELINE
# ============================================================

st.divider()
st.header("3. Process Resumes and Rank Candidates")

st.write(
    "The pipeline runs PDF extraction, skill extraction, profile extraction, "
    "job-description parsing, candidate matching, and RAG database creation."
)

if st.button("Run Resume Analysis", type="primary", width='stretch' ):
    show_project_structure_warning()

    if not list(RESUME_DIR.glob("*.pdf")):
        st.error("Please upload and save at least one PDF resume first.")
    elif not JOB_DESCRIPTION_FILE.exists():
        st.error("Please save a job description before running the analysis.")
    else:
        logs_placeholder = st.container()

        with st.spinner("Processing resumes. This may take a few minutes..."):
            success, pipeline_logs = run_resume_pipeline()

        with logs_placeholder:
            st.subheader("Pipeline logs")
            for script_name, output in pipeline_logs:
                with st.expander(script_name, expanded=not success):
                    st.code(output or "No output returned.", language="text")

        if success:
            st.success("Resume analysis pipeline completed.")
        else:
            st.error(
                "The pipeline stopped because one of the steps failed. "
                "Open the relevant log above and check the error."
            )


# ============================================================
# 7. CANDIDATE RANKING RESULTS
# ============================================================

st.divider()
st.header("4. Candidate Ranking")

ranking_df = load_ranking_data()

if ranking_df.empty:
    st.info(
        "No candidate ranking results are available yet. "
        "Save resumes and a job description, then run Resume Analysis."
    )
else:
    score_column = find_score_column(ranking_df)

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Total candidates", len(ranking_df))

    if score_column:
        numeric_scores = pd.to_numeric(ranking_df[score_column], errors="coerce")
        average_score = numeric_scores.mean()
        metric_col2.metric(
            "Average matching score",
            f"{average_score:.2f}%" if pd.notna(average_score) else "N/A",
        )
    else:
        metric_col2.metric("Average matching score", "Score column not found")

    st.dataframe(ranking_df, width='stretch' , hide_index=True)

    if score_column:
        chart_df = ranking_df.copy()
        chart_df[score_column] = pd.to_numeric(
            chart_df[score_column], errors="coerce"
        )
        chart_df = chart_df.dropna(subset=[score_column])

        if not chart_df.empty:
            # Use the first available candidate-identifying column as chart labels.
            label_candidates = [
                "candidate_name",
                "name",
                "resume_name",
                "filename",
                "file_name",
            ]
            lower_to_original = {
                str(col).lower(): col for col in chart_df.columns
            }
            label_column = next(
                (
                    lower_to_original[name]
                    for name in label_candidates
                    if name in lower_to_original
                ),
                None,
            )

            if label_column:
                chart_data = chart_df.set_index(label_column)[[score_column]]
                st.subheader("Candidate matching scores")
                st.bar_chart(chart_data)

    csv_data = ranking_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Candidate Ranking CSV",
        data=csv_data,
        file_name="candidate_ranking.csv",
        mime="text/csv",
    )


# ============================================================
# 8. AI RECRUITER ASSISTANT
# ============================================================

st.divider()
st.header("5. AI Recruiter Assistant")

st.write(
    "Ask questions about uploaded resumes, candidate skills, experience, "
    "education, or candidate-job matching."
)

if "recruiter_chat_history" not in st.session_state:
    st.session_state.recruiter_chat_history = []

if "recruiter_question" not in st.session_state:
    st.session_state.recruiter_question = ""

# Display previous conversation.
for message in st.session_state.recruiter_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Example: Which candidate has Python and NLP experience?"
)

if question:
    st.session_state.recruiter_chat_history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching resume knowledge and generating an answer..."):
            try:
                # Pass only the earlier conversation as history.
                previous_history = st.session_state.recruiter_chat_history[:-1]
                answer = get_rag_answer(question, previous_history)
                st.markdown(answer)

                st.session_state.recruiter_chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except ImportError as exc:
                answer = (
                    "RAG module could not be imported. Check that src is lowercase, "
                    "src/__init__.py exists, and all required packages are installed.\n\n"
                    f"Technical error: {exc}"
                )
                st.error(answer)
                st.session_state.recruiter_chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                answer = (
                    "I could not generate an answer. Check that Ollama is running, "
                    "the configured model is installed, and the ChromaDB database "
                    "has been created.\n\n"
                    f"Technical error: {exc}"
                )
                st.error(answer)
                st.session_state.recruiter_chat_history.append(
                    {"role": "assistant", "content": answer}
                )

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear Chat"):
        st.session_state.recruiter_chat_history = []
        st.rerun()


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()
st.caption(
    "Smart Resume Intelligence System | NLP • Semantic Search • "
    "Resume Matching • Retrieval-Augmented Generation"
)
