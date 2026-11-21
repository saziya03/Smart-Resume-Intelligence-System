

"""

Smart Resume Intelligence System

Universal Resume Screening + AI Recruiter Assistant



Run:

    streamlit run app/streamlit_app.py

"""



from __future__ import annotations



import sys

import os

import stat

import json

import shutil

import subprocess

import inspect

import logging

from pathlib import Path

from typing import Any



import pandas as pd

import streamlit as st





# ============================================================

# 1. PROJECT CONFIGURATION

# ============================================================



APP_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = APP_DIR.parent



if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



SRC_DIR = PROJECT_ROOT / "src"



RESUME_DIR = PROJECT_ROOT / "data" / "current_uploads"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_text"

PROFILE_DIR = PROJECT_ROOT / "data" / "extracted_profiles"



JOB_DESCRIPTION_DIR = PROJECT_ROOT / "data" / "job_descriptions"

PROCESSED_JD_DIR = PROJECT_ROOT / "data" / "processed_jds"



RESULT_DIR = PROJECT_ROOT / "data" / "matching_results"



MODEL_DIR = PROJECT_ROOT / "models"

VECTOR_DB_DIR = MODEL_DIR / "resume_vector_db"



CURRENT_JD_FILE = JOB_DESCRIPTION_DIR / "current_job.txt"

CURRENT_JD_JSON = PROCESSED_JD_DIR / "current_job.json"



RANKING_FILE = RESULT_DIR / "candidate_ranking.csv"



for folder in [

    RESUME_DIR,

    PROCESSED_DIR,

    PROFILE_DIR,

    JOB_DESCRIPTION_DIR,

    PROCESSED_JD_DIR,

    RESULT_DIR,

    MODEL_DIR,

]:

    folder.mkdir(parents=True, exist_ok=True)



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)





# ============================================================

# 2. PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="Smart Resume Intelligence System",

    page_icon="📄",

    layout="wide",

    initial_sidebar_state="expanded",

)



st.title("📄 Smart Resume Intelligence System")



st.caption(

    "Universal Resume Screening | NLP | Semantic Matching | "

    "Llama | RAG | AI Recruiter Assistant"

)





# ============================================================

# 3. SESSION STATE

# ============================================================



DEFAULT_STATE = {

    "recruiter_chat_history": [],

    "analysis_completed": False,

    "selected_job": "current_job",

}



for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value





# ============================================================

# 4. HELPER FUNCTIONS

# ============================================================





def clear_folder_contents(folder: Path):

    """

    Delete folder contents without deleting the main folder.

    Handles Windows read-only permissions where possible.

    """



    folder = Path(folder)



    folder.mkdir(parents=True, exist_ok=True)



    def handle_remove_error(func, path, exc):

        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)

        func(path)



    for item in folder.iterdir():



        if item.is_dir() and not item.is_symlink():



            shutil.rmtree(

                item,

                onexc=handle_remove_error

            )



        else:



            try:

                item.unlink()



            except PermissionError:

                os.chmod(

                    item,

                    stat.S_IWRITE | stat.S_IREAD

                )

                item.unlink()


# Clear temporary application data once when a NEW Streamlit session starts.
# Original sample files in data/resumes are intentionally preserved.
if not st.session_state.get("_session_data_initialized", False):
    temporary_folders = [
        RESUME_DIR,
        PROCESSED_DIR,
        PROFILE_DIR,
        JOB_DESCRIPTION_DIR,
        PROCESSED_JD_DIR,
        RESULT_DIR,
        VECTOR_DB_DIR,
    ]

    for folder in temporary_folders:
        clear_folder_contents(folder)

    for file_path in [CURRENT_JD_FILE, CURRENT_JD_JSON]:
        if file_path.exists():
            file_path.unlink()

    st.session_state["recruiter_chat_history"] = []
    st.session_state["analysis_completed"] = False
    st.session_state["selected_job"] = "current_job"
    st.session_state["_session_data_initialized"] = True


def load_json(file_path: Path) -> dict:



    if not file_path.exists():

        return {}



    try:

        with open(file_path, "r", encoding="utf-8") as file:

            return json.load(file)



    except Exception as exc:

        logger.warning("JSON loading error: %s", exc)

        return {}





def load_ranking_data() -> pd.DataFrame:



    if not RANKING_FILE.exists():

        return pd.DataFrame()



    try:

        return pd.read_csv(RANKING_FILE)



    except pd.errors.EmptyDataError:

        return pd.DataFrame()



    except Exception as exc:

        st.error(f"Could not load ranking results: {exc}")

        return pd.DataFrame()





def run_python_script(script_name: str):



    script_path = SRC_DIR / script_name



    if not script_path.exists():

        return False, f"File not found: {script_path}"



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



        output = result.stdout or ""

        errors = result.stderr or ""



        if result.returncode != 0:



            return False, (

                f"Script failed: {script_name}\n\n"

                f"Output:\n{output}\n\n"

                f"Error:\n{errors}"

            )



        return True, output + ("\n" + errors if errors else "")



    except subprocess.TimeoutExpired:

        return False, f"{script_name} exceeded the 10-minute limit."



    except Exception as exc:

        return False, str(exc)





# ============================================================

# 5. RESUME UPLOAD

# ============================================================



def save_uploaded_resumes(uploaded_files) -> _int_:



    saved = 0



    for uploaded_file in uploaded_files:



        if not uploaded_file.name.lower().endswith(".pdf"):

            st.warning(f"Skipped: {uploaded_file.name}")

            continue



        safe_name = Path(uploaded_file.name).name



        destination = RESUME_DIR / safe_name



        try:

            destination.write_bytes(uploaded_file.getvalue())

            saved += 1



        except OSError as exc:

            st.error(f"Could not save {safe_name}: {exc}")



    return saved





# ============================================================

# 6. SAVE UNIVERSAL JOB DESCRIPTION

# ============================================================



def save_job_description(job_title: _str_, description: str):



    content = (

        f"Job Title: {job_title.strip()}\n\n"

        f"Job Description:\n"

        f"{description.strip()}"

    )



    CURRENT_JD_FILE.write_text(

        content,

        encoding="utf-8"

    )



    return CURRENT_JD_FILE





# ============================================================

# 7. RAG DATABASE UPDATE

# ============================================================



def update_rag_database():



    try:



        from src.rag_engine import store_resumes



        store_resumes()



        return True, "Resume knowledge database updated."



    except Exception as exc:



        return False, (

            "RAG database update failed.\n"

            f"Technical details: {exc}"

        )





# ============================================================

# 8. COMPLETE RESUME PIPELINE

# ============================================================



def run_resume_pipeline():



    pipeline_scripts = [

        "pdf_parser.py",

        "skill_extractor.py",

        "profile_extractor.py",

        "job_description_parser.py",

        "matching_engine.py",

    ]



    logs = []



    for script_name in pipeline_scripts:



        success, output = run_python_script(script_name)



        logs.append((script_name, output))



        if not success:

            return False, logs



    # Verify the current JD was actually parsed.

    if not CURRENT_JD_JSON.exists():



        return False, logs + [

            (

                "JD validation",

                "current_job.json was not created. "

                "Check job_description_parser.py."

            )

        ]



    # Matching results must exist.

    if not RANKING_FILE.exists():



        return False, logs + [

            (

                "Matching validation",

                "candidate_ranking.csv was not generated."

            )

        ]



    # RAG is supplementary; it should not erase

    # successfully generated matching results.

    rag_success, rag_message = update_rag_database()



    logs.append(("RAG database", rag_message))



    if not rag_success:

        logs.append((

            "RAG status",

            "Candidate ranking is available, but chatbot "

            "may not have updated resume information."

        ))



    return True, logs





# ============================================================

# 9. CURRENT JOB FILTER

# ============================================================



def get_current_job_results(df: pd.DataFrame):



    if df.empty:

        return df



    if "job_id" not in df.columns:

        return df



    return df[

        df["job_id"].astype(str) == "current_job"

    ].copy()





# ============================================================

# 10. PROFILE DETAILS

# ============================================================



def get_candidate_profile(candidate_name):



    profile_path = PROFILE_DIR / f"{candidate_name}.json"



    return load_json(profile_path)





def show_candidate_details(candidate_name):



    profile = get_candidate_profile(candidate_name)



    if not profile:

        st.info("Detailed candidate profile is not available.")

        return



    st.subheader("Candidate Profile")



    st.write(

        "**Name:**",

        profile.get("name", candidate_name)

    )



    for field in [

        "email",

        "phone",

        "education",

        "experience",

        "total_experience",

        "technical_skills",

        "soft_skills",

        "projects",

        "certifications",

    ]:



        value = profile.get(field)



        if value:

            st.write(f"**{field.replace('_', ' ').title()}:**")



            if isinstance(value, (list, dict)):

                st.json(value)

            else:

                st.write(value)





# ============================================================

# 11. FLEXIBLE RAG FUNCTION

# ============================================================



def get_rag_answer(question: _str_, chat_history: list):



    from src.rag_engine import generate_answer



    # Inspect the actual function signature.

    # This avoids passing unsupported chat_history/top_k

    # parameters to an older version of rag_engine.py.



    signature = inspect.signature(generate_answer)



    parameters = signature.parameters



    accepts_kwargs = any(

        parameter.kind == inspect.Parameter.VAR_KEYWORD

        for parameter in parameters.values()

    )



    available_kwargs = {

        "question": question,

        "chat_history": chat_history,

        "top_k": 5,

    }



    kwargs = {

        key: value

        for key, value in available_kwargs.items()

        if key in parameters or accepts_kwargs

    }



    if "question" in kwargs:

        answer = generate_answer(**kwargs)



    elif "query" in parameters:

        kwargs["query"] = question

        answer = generate_answer(**{

            key: value

            for key, value in kwargs.items()

            if key in parameters or accepts_kwargs

        })



    else:

        answer = generate_answer(question)



    if answer is None:

        return "The AI assistant returned an empty response."



    if isinstance(answer, dict):



        return (

            answer.get("answer")

            or answer.get("response")

            or answer.get("result")

            or str(answer)

        )



    return str(answer)





# ============================================================

# 12. SIDEBAR

# ============================================================



with st.sidebar:



    st.header("⚙️ Project Status")



    st.caption("Project directory")



    st.code(str(PROJECT_ROOT))



    st.divider()



    st.metric(

        "Uploaded PDF Resumes",

        len(list(RESUME_DIR.glob("*.pdf")))

    )



    st.metric(

        "Processed Resumes",

        len(list(PROCESSED_DIR.glob("*.txt")))

    )



    st.metric(

        "Extracted Profiles",

        len(list(PROFILE_DIR.glob("*.json")))

    )



    st.metric(

        "Available Job Descriptions",

        len(list(JOB_DESCRIPTION_DIR.glob("*.txt")))

    )



    st.divider()



    if st.button("🔄 Refresh Application", use_container_width=True):

        st.rerun()



    st.divider()





    st.subheader("Session Management")



    if st.button(

        "🗑️ Clear All Project Data",

        use_container_width=True

    ):

        st.session_state.confirm_clear = True



    if st.session_state.get("confirm_clear", False):



        st.warning(

            "This will permanently delete uploaded resumes, "

            "processed data, profiles, current job description, "

            "matching results and vector database."

        )



        confirm_col1, confirm_col2 = st.columns(2)



        with confirm_col1:



            if st.button("Confirm Delete", type="primary"):



                folders = [

                    RESUME_DIR,

                    PROCESSED_DIR,

                    PROFILE_DIR,

                    PROCESSED_JD_DIR,

                    RESULT_DIR,

                    VECTOR_DB_DIR,

                ]



                try:



                    for folder in folders:

                        clear_folder_contents(folder)



                    # Remove only current job description.

                    if CURRENT_JD_FILE.exists():

                        CURRENT_JD_FILE.unlink()



                    # Reset application session.

                    st.session_state.recruiter_chat_history = []

                    st.session_state.analysis_completed = False

                    st.session_state.selected_job = "current_job"

                    st.session_state.confirm_clear = False



                    st.success(

                        "All project data cleared successfully!"

                    )



                    st.rerun()



                except PermissionError as exc:



                    st.error(

                        f"Windows permission error: {exc}"

                    )



                    st.warning(

                        "Close any process using these files, "

                        "check OneDrive sync, and try again."

                    )



                except Exception as exc:



                    st.error(

                        f"Data clearing failed: {exc}"

                    )



        with confirm_col2:



            if st.button("Cancel"):



                st.session_state.confirm_clear = False

                st.rerun()



# ============================================================

# 13. RESUME UPLOAD SECTION

# ============================================================



st.header("1. Upload Candidate Resumes")



st.write(

    "Upload multiple candidate PDF resumes. "

    "The system will extract and analyze candidate information."

)



uploaded_files = st.file_uploader(

    "Select candidate PDF resumes",

    type=["pdf"],

    accept_multiple_files=True,

    help="Multiple PDF resume uploads are supported."

)



if uploaded_files:



    st.info(f"{len(uploaded_files)} PDF files selected.")



    if st.button(

        "Save Uploaded Resumes",

        type="primary",

        use_container_width=True

    ):



        count = save_uploaded_resumes(uploaded_files)



        st.success(f"{count} resumes saved successfully.")





# ============================================================

# 14. UNIVERSAL JOB SEARCH / JD INPUT

# ============================================================



st.divider()



st.header("2. Universal Job Matching")



st.write(

    "Enter any job designation and paste its job description. "

    "The system will compare all uploaded resumes against this role."

)



job_title = st.text_input(

    "Search / Enter Any Job Designation",

    placeholder="Example: NLP Engineer, Data Analyst, HR Manager, Cloud Architect"

)



job_description = st.text_area(

    "Paste Complete Job Description",

    height=250,

    placeholder=(

        "Enter job responsibilities, required skills, "

        "qualifications, experience and preferred skills..."

    )

)



if st.button(

    "💾 Save Current Job Description",

    use_container_width=True

):



    if not job_title.strip():

        st.error("Please enter a job designation.")



    elif not job_description.strip():

        st.error("Please enter the job description.")



    else:



        save_job_description(

            job_title,

            job_description

        )



        st.session_state.selected_job = "current_job"



        st.success(

            f"Job description saved for: {job_title}"

        )





if CURRENT_JD_FILE.exists():



    st.success(

        "Current job description is saved and ready for analysis."

    )





# ============================================================

# 15. RUN ANALYSIS

# ============================================================



st.divider()



st.header("3. Resume Analysis & Candidate Ranking")



st.write(

    "This pipeline runs PDF extraction, NLP skill extraction, "

    "candidate profile extraction, JD parsing and semantic matching."

)



if st.button(

    "🚀 Run Universal Resume Analysis",

    type="primary",

    use_container_width=True

):



    pdf_files = list(RESUME_DIR.glob("*.pdf"))



    if not pdf_files:



        st.error("Please upload at least one PDF resume.")



    elif not CURRENT_JD_FILE.exists():



        st.error("Please save a job description first.")



    else:



        logs = []



        with st.spinner(

            "Analyzing resumes and matching candidates..."

        ):



            success, logs = run_resume_pipeline()



        st.subheader("Pipeline Execution Logs")



        for script_name, output in logs:



            with st.expander(script_name):



                st.code(

                    output or "Completed without console output.",

                    language="text"

                )



        if success:



            st.session_state.analysis_completed = True



            st.success(

                "Resume analysis and candidate matching completed."

            )



        else:



            st.error(

                "Pipeline stopped. Open the relevant log "

                "to identify the error."

            )





# ============================================================

# 16. CANDIDATE RANKING DASHBOARD

# ============================================================



st.divider()



st.header("4. Candidate Matching Dashboard")



all_ranking_df = load_ranking_data()



ranking_df = get_current_job_results(all_ranking_df)



current_job_data = load_json(CURRENT_JD_JSON)



if current_job_data:



    st.caption(

        f"Current role: {current_job_data.get('job_title', 'Not specified')}"

    )



if ranking_df.empty:



    st.info(

        "No results available for the current job. "

        "Save a job description and run analysis."

    )



else:



    if "final_score" in ranking_df.columns:



        ranking_df["final_score"] = pd.to_numeric(

            ranking_df["final_score"],

            errors="coerce"

        )



        ranking_df = ranking_df.sort_values(

            "final_score",

            ascending=False

        )



    total_candidates = len(ranking_df)



    average_score = (

        ranking_df["final_score"].mean()

        if "final_score" in ranking_df.columns

        else 0

    )



    top_score = (

        ranking_df["final_score"].max()

        if "final_score" in ranking_df.columns

        else 0

    )



    col1, col2, col3 = st.columns(3)



    col1.metric("Total Candidates", total_candidates)



    col2.metric(

        "Average Match",

        f"{average_score:.2f}%"

    )



    col3.metric(

        "Highest Match",

        f"{top_score:.2f}%"

    )



    st.subheader("Candidate Ranking")



    display_columns = [

        "candidate",

        "job_title",

        "final_score",

        "recommendation",

        "matched_skills",

        "missing_skills",

    ]



    display_columns = [

        col for col in display_columns

        if col in ranking_df.columns

    ]



    st.dataframe(

        ranking_df[display_columns],

        use_container_width=True,

        hide_index=True

    )



    # Recommendation counts

    if "recommendation" in ranking_df.columns:



        st.subheader("Candidate Recommendations")



        recommendation_counts = (

            ranking_df["recommendation"]

            .value_counts()

        )



        st.bar_chart(recommendation_counts)



    # Candidate matching scores

    if "candidate" in ranking_df.columns and "final_score" in ranking_df.columns:



        st.subheader("Candidate Match Score")



        chart_df = ranking_df.set_index("candidate")[

            ["final_score"]

        ]



        st.bar_chart(chart_df)



    # Candidate selection

    st.divider()



    st.subheader("Detailed Candidate Analysis")



    candidate_list = ranking_df["candidate"].astype(str).tolist()



    selected_candidate = st.selectbox(

        "Select Candidate",

        candidate_list

    )



    candidate_row = ranking_df[

        ranking_df["candidate"].astype(str) == selected_candidate

    ].iloc[0]



    score = candidate_row.get("final_score", 0)



    st.write(

        f"\\Candidate:\\ {selected_candidate}"

    )



    st.metric(

        "Overall Match Score",

        f"{float(score):.2f}%"

    )



    st.write(

        "**Recommendation:**",

        candidate_row.get("recommendation", "Not available")

    )



    col1, col2 = st.columns(2)



    with col1:



        st.subheader("✅ Matched Skills")



        st.write(

            candidate_row.get("matched_skills", "Not available")

            or "No matched skills found."

        )



        st.subheader("⭐ Matched Preferred Skills")



        st.write(

            candidate_row.get("matched_preferred_skills", "")

            or "No preferred skills matched."

        )



    with col2:



        st.subheader("❌ Missing Required Skills")



        st.write(

            candidate_row.get("missing_skills", "")

            or "No missing required skills."

        )



        st.subheader("Additional Skills Gap")



        st.write(

            candidate_row.get("missing_preferred_skills", "")

            or "No missing preferred skills."

        )



    with st.expander("View Full Candidate Profile"):



        show_candidate_details(selected_candidate)



    st.divider()



    csv_data = ranking_df.to_csv(

        index=False

    ).encode("utf-8-sig")



    st.download_button(

        "⬇️ Download Current Job Ranking",

        data=csv_data,

        file_name="current_job_candidate_ranking.csv",

        mime="text/csv",

        use_container_width=True

    )



    # Show all jobs if available

    if not all_ranking_df.empty and "job_title" in all_ranking_df.columns:



        with st.expander("View Results Across All Job Descriptions"):



            st.dataframe(

                all_ranking_df,

                use_container_width=True,

                hide_index=True

            )



            all_csv = all_ranking_df.to_csv(

                index=False

            ).encode("utf-8-sig")



            st.download_button(

                "Download All Job Matching Results",

                data=all_csv,

                file_name="all_job_candidate_ranking.csv",

                mime="text/csv"

            )





# ============================================================

# 17. AI RECRUITER ASSISTANT

# ============================================================



st.divider()



st.header("5. AI Recruiter Assistant")



st.write(

    "Ask questions about candidates, skills, education, "

    "experience, projects, job matching or general technical topics."

)



st.info(

    "Powered by your Llama + RAG pipeline, "

    "provided rag_engine.py supports these operations."

)



# Display chat history

for message in st.session_state.recruiter_chat_history:



    with st.chat_message(message["role"]):

        st.markdown(message["content"])





question = st.chat_input(

    "Ask anything: Who is the best NLP Engineer candidate?"

)



if question:



    previous_history = list(

        st.session_state.recruiter_chat_history

    )



    st.session_state.recruiter_chat_history.append({

        "role": "user",

        "content": question

    })



    with st.chat_message("user"):

        st.markdown(question)



    with st.chat_message("assistant"):



        with st.spinner(

            "Searching candidate knowledge and generating response..."

        ):



            try:



                answer = get_rag_answer(

                    question,

                    previous_history

                )



                st.markdown(answer)



            except Exception as exc:



                answer = (

                    "Unable to generate an answer.\n\n"

                    "Please verify that Ollama is running, "

                    "the configured Llama model is installed, "

                    "and the RAG database is available.\n\n"

                    f"Technical error: {exc}"

                )



                st.error(answer)



            st.session_state.recruiter_chat_history.append({

                "role": "assistant",

                "content": answer

            })





chat_col1, chat_col2 = st.columns([1, 5])



with chat_col1:



    if st.button("🗑️ Clear Chat"):



        st.session_state.recruiter_chat_history = []



        st.rerun()





# ============================================================

# 18. FOOTER

# ============================================================



st.divider()



st.caption(

    "Smart Resume Intelligence System | "

    "Python • NLP • Sentence Transformers • ChromaDB • "

    "RAG • Llama • Streamlit"

)
