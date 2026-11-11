import sys
import subprocess
import streamlit as st
import pandas as pd

from pathlib import Path


# =========================================================
# 1. PROJECT PATHS
# =========================================================

# This assumes this file is inside the app/ folder.
# Example:
# Smart-Resume-Intelligence-System/
#   app/streamlit_app.py
#   Src/rag_engine.py
#   data/

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "Src"

RESUME_DIR = BASE_DIR / "data" / "resumes"
JD_DIR = BASE_DIR / "data" / "job_descriptions"
RESULT_FILE = BASE_DIR / "data" / "matching_results" / "candidate_ranking.csv"

# Make the project root available for importing Src.rag_engine
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.rag_engine import (
        generate_answer as rag_generate_answer,
        store_resumes,
    )
    RAG_IMPORT_ERROR = None
except Exception as error:
    rag_generate_answer = None
    store_resumes = None
    RAG_IMPORT_ERROR = str(error)


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def generate_answer(question):
    """Generate an answer using the RAG engine and chat history."""

    if rag_generate_answer is None:
        return (
            "The RAG engine could not be imported.\n\n"
            f"Import error: {RAG_IMPORT_ERROR}\n\n"
            "Please check the Src folder name and rag_engine.py file."
        )

    # The current question is already saved in chat_history.
    # Send only earlier messages as conversation history.
    previous_history = st.session_state.chat_history[:-1]

    return rag_generate_answer(
        question=question,
        chat_history=previous_history,
        top_k=5,
    )


def save_uploaded_resumes(uploaded_files):
    """Save uploaded PDF files into data/resumes."""

    RESUME_DIR.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        safe_filename = Path(uploaded_file.name).name
        file_path = RESUME_DIR / safe_filename

        file_path.write_bytes(uploaded_file.getvalue())


def save_job_description(job_title, job_description):
    """Save the current job description as a text file."""

    JD_DIR.mkdir(parents=True, exist_ok=True)

    jd_text = (
        f"Job Title: {job_title.strip()}\n\n"
        f"{job_description.strip()}"
    )

    jd_file = JD_DIR / "current_job.txt"
    jd_file.write_text(jd_text, encoding="utf-8")


def run_resume_pipeline():
    """Run all existing resume-processing scripts in sequence."""

    pipeline_steps = [
        ("PDF Parsing", "pdf_parser.py"),
        ("Skill Extraction", "skill_extractor.py"),
        ("Profile Extraction", "profile_extractor.py"),
        ("Job Description Analysis", "job_description_parser.py"),
        ("Resume Matching", "matching_engine.py"),
    ]

    if not SRC_DIR.exists():
        st.error(f"Source folder not found: {SRC_DIR}")
        st.info("Check that your folder is named exactly 'Src'.")
        return False

    progress = st.progress(0)
    status = st.empty()

    for index, (step_name, script_name) in enumerate(pipeline_steps):
        script_path = SRC_DIR / script_name

        if not script_path.exists():
            st.error(f"Script not found: {script_path}")
            return False

        status.write(f"Running: **{step_name}**")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )

            with st.expander(f"{step_name} output"):
                st.code(result.stdout or "No standard output.")

                if result.stderr:
                    st.text("Error/output details:")
                    st.code(result.stderr)

            if result.returncode != 0:
                st.error(f"{step_name} failed. Check the output above.")
                return False

        except subprocess.TimeoutExpired:
            st.error(f"{step_name} timed out. Please try again.")
            return False

        except Exception as error:
            st.error(f"{step_name} failed: {error}")
            return False

        progress.progress((index + 1) / len(pipeline_steps))

    status.success("All resume-processing steps completed.")
    return True


# =========================================================
# 3. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Resume Intelligence",
    page_icon="📄",
    layout="wide",
)


# =========================================================
# 4. CUSTOM HEADER
# =========================================================

st.title("📄 Smart Resume Intelligence System")

st.write(
    "AI-powered resume analysis and candidate-job matching platform."
)

st.divider()


# =========================================================
# 5. RESUME UPLOAD
# =========================================================

st.header("1. Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)


# =========================================================
# 6. JOB DESCRIPTION INPUT
# =========================================================

st.header("2. Enter Job Description")

job_title = st.text_input(
    "Job Title",
    placeholder="Example: NLP Engineer",
)

job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Enter the job requirements and responsibilities...",
)


# =========================================================
# 7. PROCESS RESUMES AND MATCH
# =========================================================

if st.button("Analyze Resumes", type="primary"):

    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")

    elif not job_title.strip() or not job_description.strip():
        st.error("Please enter the job title and job description.")

    else:
        # Avoid displaying old results if the new analysis fails.
        st.session_state.pop("results", None)

        with st.spinner("Saving uploaded files..."):
            save_uploaded_resumes(uploaded_files)
            save_job_description(job_title, job_description)

        st.success("Uploaded resumes and job description saved.")

        pipeline_success = run_resume_pipeline()

        if pipeline_success:

            if not RESULT_FILE.exists():
                st.error(
                    "Matching CSV was not generated. "
                    "Please check matching_engine.py."
                )

            else:
                try:
                    results = pd.read_csv(RESULT_FILE)

                    required_columns = {"candidate", "final_score"}

                    if not required_columns.issubset(results.columns):
                        st.error(
                            "The candidate_ranking.csv file must contain "
                            "'candidate' and 'final_score' columns."
                        )
                    else:
                        st.session_state["results"] = results
                        st.success("Resume analysis completed!")

                        # Build/update the vector database for the chatbot.
                        if store_resumes is not None:
                            with st.spinner(
                                "Updating resume knowledge database..."
                            ):
                                total_chunks = store_resumes()

                            st.success(
                                "Resume knowledge database updated. "
                                f"Chunks stored: {total_chunks}"
                            )
                        else:
                            st.warning(
                                "RAG engine is unavailable. "
                                "The chatbot cannot search resumes yet."
                            )

                except Exception as error:
                    st.error(f"Could not read candidate results: {error}")


# =========================================================
# 8. DISPLAY CANDIDATE RESULTS
# =========================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    st.divider()
    st.header("3. Candidate Ranking")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Candidates",
            len(results),
        )

    with col2:
        average_score = (
            results["final_score"].mean()
            if not results.empty
            else 0
        )

        st.metric(
            "Average Match Score",
            f"{average_score:.2f}%",
        )

    st.subheader("Candidate Matching Results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    if not results.empty:
        st.subheader("Candidate Match Score")

        chart_data = results.set_index("candidate")["final_score"]
        st.bar_chart(chart_data)

    st.download_button(
        label="Download Candidate Results",
        data=results.to_csv(index=False),
        file_name="candidate_ranking.csv",
        mime="text/csv",
    )


# =========================================================
# 9. FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Resume Intelligence System | "
    "NLP + Sentence Transformers + ChromaDB + AI"
)


# =========================================================
# 10. AI RECRUITER ASSISTANT
# =========================================================

st.divider()
st.header("4. AI Recruiter Assistant")

st.write(
    "Ask questions about candidates, skills, education, experience, "
    "projects, job matching, or general technical topics."
)

if RAG_IMPORT_ERROR:
    st.warning(
        "RAG engine import issue: "
        f"{RAG_IMPORT_ERROR}"
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous conversation.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask something about the resumes or a technical topic..."
)

if question:

    # Display and save the current user question.
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Generate the AI response.
    with st.chat_message("assistant"):
        with st.spinner("Searching resumes and generating answer..."):

            try:
                answer = generate_answer(question)
            except Exception as error:
                answer = f"An error occurred while generating the answer: {error}"

            st.markdown(answer)

    # Save the assistant response for future conversation context.
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

# Clear chat without deleting uploaded resumes or analysis results.
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
