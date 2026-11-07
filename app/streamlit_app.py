import streamlit as st
import subprocess
import sys
import pandas as pd
import json

from pathlib import Path

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "resumes"
JD_DIR = BASE_DIR / "data" / "job_descriptions"

RESULT_FILE = (
    BASE_DIR / "data" / "matching_results"
    / "candidate_ranking.csv"
)

SRC_DIR = BASE_DIR / "src"

# =====================================
# CHATBOT HELPER
# =====================================

def generate_answer(question):
    q = question.lower()

    if "results" not in st.session_state:
        return "Please first upload resumes and click on Analyze Resumes, then ask me about candidates."

    results = st.session_state["results"]

    if "best" in q or "top" in q or "highest" in q or "sabse achha" in q:
        top = results.sort_values("final_score", ascending=False).iloc[0]
        return f"**Top candidate is {top['candidate']}** with a match score of {top['final_score']:.2f}%."

    if "how many" in q or "kitne" in q or "total" in q:
        return f"Total **{len(results)}** candidates were analyzed. Average score is {results['final_score'].mean():.2f}%."

    for _, row in results.iterrows():
        name = str(row['candidate']).lower()
        first_name = name.split()[0]
        if first_name in q and len(first_name) > 2:
            profile_dir = BASE_DIR / "data" / "profiles"
            skills_text = "Not found"
            if profile_dir.exists():
                for f in profile_dir.glob("*.json"):
                    if first_name in f.name.lower():
                        try:
                            data = json.loads(f.read_text(encoding="utf-8"))
                            skills = data.get("skills", [])
                            skills_text = ", ".join(skills) if skills else "Not found"
                        except:
                            pass
                        break
            return f"**{row['candidate']}**\n\n- Match Score: {row['final_score']:.2f}%\n- Skills: {skills_text}"

    if "who" in q and "know" in q:
        return f"Here is the full ranking:\n\n{results.to_string(index=False)}"

    return f"Here are the candidates ranked by match score:\n\n{results.to_string(index=False)}"

# =====================================
# 2. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Resume Intelligence",
    page_icon="📄",
    layout="wide"
)

# =====================================
# 3. CUSTOM HEADER
# =====================================

st.title("📄 Smart Resume Intelligence System")

st.write(
    "AI-powered resume analysis and "
    "candidate-job matching platform."
)

st.divider()

# =====================================
# 4. RESUME UPLOAD
# =====================================

st.header("1. Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# =====================================
# 5. JOB DESCRIPTION INPUT
# =====================================

st.header("2. Enter Job Description")

job_title = st.text_input(
    "Job Title",
    placeholder="Example: NLP Engineer"
)

job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Enter the job requirements and responsibilities..."
)

# =====================================
# 6. PROCESS RESUMES AND MATCH
# =====================================

if st.button("Analyze Resumes", type="primary"):

    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")

    elif not job_title.strip() or not job_description.strip():
        st.error("Please enter the job title and description.")

    else:

        RESUME_DIR.mkdir(parents=True, exist_ok=True)
        JD_DIR.mkdir(parents=True, exist_ok=True)

        # Save uploaded resumes
        for uploaded_file in uploaded_files:

            file_path = RESUME_DIR / Path(
                uploaded_file.name
            ).name

            file_path.write_bytes(
                uploaded_file.getvalue()
            )

        # Save job description
        jd_file = JD_DIR / "current_job.txt"

        jd_text = (
            f"Job Title: {job_title}\n\n"
            f"{job_description}"
        )

        jd_file.write_text(
            jd_text,
            encoding="utf-8"
        )

        # Run existing Python pipeline
        pipeline_steps = [
            ("PDF Parsing", "pdf_parser.py"),
            ("Skill Extraction", "skill_extractor.py"),
            ("Profile Extraction", "profile_extractor.py"),
            ("Job Description Analysis", "job_description_parser.py"),
            ("Resume Matching", "matching_engine.py")
        ]

        pipeline_success = True

        progress = st.progress(0)

        for index, (step_name, script_name) in enumerate(
            pipeline_steps
        ):

            st.write(f"Running: {step_name}")

            script_path = SRC_DIR / script_name

            try:

                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if result.returncode != 0:

                    st.error(f"{step_name} failed.")

                    st.code(
                        result.stderr or result.stdout
                    )

                    pipeline_success = False
                    break

                with st.expander(f"{step_name} output"):
                    st.code(result.stdout or "Completed.")

            except subprocess.TimeoutExpired:

                st.error(
                    f"{step_name} timed out. "
                    "Please try again."
                )

                pipeline_success = False
                break

            except Exception as error:

                st.error(str(error))
                pipeline_success = False
                break

            progress.progress(
                (index + 1) / len(pipeline_steps)
            )

        # Display results
        if pipeline_success:

            st.success("Resume analysis completed!")

            if RESULT_FILE.exists():

                results = pd.read_csv(RESULT_FILE)

                st.session_state["results"] = results

            else:

                st.warning(
                    "Matching CSV was not generated. "
                    "Check your matching engine."
                )

# =====================================
# 7. DISPLAY CANDIDATE RESULTS
# =====================================

if "results" in st.session_state:

    results = st.session_state["results"]

    st.divider()

    st.header("3. Candidate Ranking")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Candidates",
            len(results)
        )

    with col2:
        st.metric(
            "Average Match Score",
            f"{results['final_score'].mean():.2f}%"
        )

    st.subheader("Candidate Matching Results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Candidate Match Score")

    chart_data = results.set_index("candidate")[
        "final_score"
    ]

    st.bar_chart(chart_data)

    st.download_button(
        label="Download Candidate Results",
        data=results.to_csv(index=False),
        file_name="candidate_ranking.csv",
        mime="text/csv"
    )

# =====================================
# 8. FOOTER
# =====================================

st.divider()

st.caption(
    "Smart Resume Intelligence System | "
    "NLP + Sentence Transformers + ChromaDB + AI"
)

# =====================================
# 9. AI RECRUITER CHATBOT
# =====================================

st.divider()

st.header("4. AI Recruiter Assistant")

st.write(
    "Ask questions about candidates using "
    "the uploaded resume information."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input(
    "Ask something about the resumes..."
)

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing resumes..."):

            try:

                answer = generate_answer(question)

                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:

                st.error(f"Error: {error}")


