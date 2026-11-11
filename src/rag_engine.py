
import chromadb
import ollama

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "processed_text"
VECTOR_DIR = BASE_DIR / "models" / "resume_vector_db"


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# ==========================================
# 3. CONNECT CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=str(VECTOR_DIR)
)

collection = client.get_or_create_collection(
    name="resume_knowledge",
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# 4. TEXT CHUNKING
# ==========================================

def create_chunks(text, chunk_size=1000, overlap=150):

    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# 5. STORE RESUMES IN VECTOR DATABASE
# ==========================================

def store_resumes():

    if not RESUME_DIR.exists():
        print(f"Resume directory not found: {RESUME_DIR}")
        return 0

    total_chunks = 0

    resume_files = list(RESUME_DIR.glob("*.txt"))

    if not resume_files:
        print("No processed resume files found.")
        return 0

    for file_path in resume_files:

        print(f"Processing: {file_path.name}")

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        chunks = create_chunks(text)

        if not chunks:
            continue

        # Remove old chunks for this resume
        collection.delete(
            where={"source": file_path.name}
        )

        embeddings = embedding_model.encode(
            chunks
        ).tolist()

        ids = [
            f"{file_path.stem}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": file_path.name,
                "chunk_number": i
            }
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(chunks)

        print(f"Stored {len(chunks)} chunks.")

    print("\n========== VECTOR DATABASE REPORT ==========")
    print(f"Total resumes: {len(resume_files)}")
    print(f"Total chunks stored: {total_chunks}")
    print(f"Database records: {collection.count()}")

    return total_chunks


# ==========================================
# 6. RETRIEVE RELEVANT RESUME INFORMATION
# ==========================================

def retrieve_context(question, top_k=5):

    if not question or not question.strip():
        return []

    if collection.count() == 0:
        return []

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_data = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved_data.append({
            "content": document,
            "source": metadata.get("source", "Unknown"),
            "similarity": round(1 - distance, 4)
        })

    return retrieved_data


# ==========================================
# 7. SEARCH RESUMES
# ==========================================

def search_resumes(question, top_k=5):

    results = retrieve_context(
        question,
        top_k
    )

    if not results:
        return "No relevant resume information found."

    output = []

    for index, item in enumerate(results, start=1):

        output.append(
            f"""
Result {index}
Resume: {item['source']}
Similarity: {item['similarity']}

{item['content']}
"""
        )

    return "\n".join(output)


# ==========================================
# 8. GENERATE AI ANSWER
# ==========================================

def generate_answer(question, chat_history=None, top_k=5):

    if not question or not question.strip():
        return "Please enter a question."

    if chat_history is None:
        chat_history = []

    # Retrieve relevant resume information
    retrieved_data = retrieve_context(
        question,
        top_k=top_k
    )

    if retrieved_data:

        resume_context = "\n\n".join(
            [
                f"Resume: {item['source']}\n"
                f"Similarity: {item['similarity']}\n"
                f"Information: {item['content']}"
                for item in retrieved_data
            ]
        )

    else:
        resume_context = (
            "No relevant resume information was found."
        )

    # ======================================
    # SYSTEM PROMPT
    # ======================================

    system_prompt = """
You are an AI Recruiter Assistant for a Smart Resume Intelligence System.

Your job is to answer recruiter questions about uploaded candidates.

IMPORTANT RULES:

1. Understand the exact question before answering.
2. Give a separate and relevant answer for every question.
3. Do not repeat the same generic answer for different questions.
4. Use the retrieved resume information for candidate-specific questions.
5. Never invent candidate skills, education, experience or projects.
6. If information is missing from resumes, clearly say:
   "This information is not available in the uploaded resumes."
7. For general technical questions, you may use your general knowledge.
8. Use simple English.
9. If the user asks about a candidate, mention the candidate's resume filename.
10. Do not mix information from different candidates.

ANSWER FORMAT:

Direct Answer:
Give the answer first.

Explanation:
Explain the answer in simple language.

Strategy:
Explain how you found the answer or how to solve the problem.

Example:
Give a relevant example when useful.

Resume Connection:
Connect the answer to uploaded resumes when relevant.
If it is a general technical question, explain that it is not based on candidate resume data.
"""

    # ======================================
    # BUILD CHAT MESSAGES
    # ======================================

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Previous conversation
    for item in chat_history[-6:]:

        if item.get("role") in ["user", "assistant"]:

            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

    # Current question and retrieved context
    user_prompt = f"""
CURRENT QUESTION:
{question}

RETRIEVED RESUME INFORMATION:
{resume_context}

Answer the current question carefully.

If it asks about a candidate, use only the retrieved resume information.
If it asks a general technical question, answer using your general knowledge.

Do not answer a different question.
"""

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    # ======================================
    # CALL OLLAMA MODEL
    # ======================================

    try:

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=messages,
            options={
                "temperature": 0.4
            }
        )

        answer = response["message"]["content"]

        return answer.strip()

    except Exception as error:

        return (
            "AI response could not be generated.\n\n"
            f"Error: {str(error)}\n\n"
            "Please check whether Ollama is running and "
            "the qwen2.5:3b model is installed."
        )


# ==========================================
# 9. TEST CHATBOT FROM TERMINAL
# ==========================================

if __name__ == "__main__":

    print("\nSmart Resume Intelligence System")
    print("Building resume vector database...")

    store_resumes()

    chat_history = []

    print("\nAI Recruiter Assistant is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("Ask your question: ").strip()

        if question.lower() == "exit":
            print("Chatbot stopped.")
            break

        answer = generate_answer(
            question,
            chat_history=chat_history
        )

        print("\nAI Assistant:")
        print(answer)
        print("\n" + "=" * 60)

        chat_history.append({
            "role": "user",
            "content": question
        })

        chat_history.append({
            "role": "assistant",
            "content": answer
        })
