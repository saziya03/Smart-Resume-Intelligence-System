
import chromadb
import ollama

from sentence_transformers import SentenceTransformer
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "processed_text"

VECTOR_DIR = BASE_DIR / "models" / "resume_vector_db"


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 3. CONNECT TO CHROMADB
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
# 5. STORE RESUMES IN CHROMADB
# ==========================================

def store_resumes():

    resume_files = list(RESUME_DIR.glob("*.txt"))

    if not resume_files:
        print("No processed resumes found!")
        return

    total_chunks = 0

    for resume_file in resume_files:

        print(f"\nProcessing: {resume_file.name}")

        text = resume_file.read_text(
            encoding="utf-8"
        )

        chunks = create_chunks(text)

        if not chunks:
            print("No readable text found!")
            continue

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        ids = [
            f"{resume_file.stem}_chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": resume_file.name,
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

        print("Chunks stored:", len(chunks))

    print("\n========== RAG DATABASE REPORT ==========")

    print("Total chunks processed:", total_chunks)

    print("Total database records:", collection.count())


# ==========================================
# 6. RETRIEVE RELEVANT RESUME INFORMATION
# ==========================================

def retrieve_context(question, top_k=3):

    if collection.count() == 0:
        return []

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count())
    )

    context = []

    for i, document in enumerate(results["documents"][0]):

        source = results["metadatas"][0][i]["source"]

        distance = results["distances"][0][i]

        context.append({
            "source": source,
            "content": document,
            "distance": distance
        })

    return context


# ==========================================
# 7. SEMANTIC SEARCH
# ==========================================

def search_resumes(question, top_k=3):

    retrieved_data = retrieve_context(question, top_k)

    if not retrieved_data:
        print("No relevant resume information found.")
        return

    print("\n========== SEARCH RESULTS ==========")

    for i, item in enumerate(retrieved_data, start=1):

        similarity = max(
            0,
            (1 - item["distance"]) * 100
        )

        print(f"\nResult {i}")

        print("Resume:", item["source"])

        print("Similarity:", round(similarity, 2), "%")

        print("Content:", item["content"][:500])


# ==========================================
# 8. GENERATE ANSWERS USING OLLAMA
# ==========================================

def generate_answer(question):

    retrieved_data = retrieve_context(question, top_k=5)

    if not retrieved_data:
        print("No relevant resume information found.")
        return

    context = ""

    for item in retrieved_data:

        context += (
            f"\nResume Source: {item['source']}\n"
            f"Resume Content: {item['content']}\n"
        )

    prompt = f"""
You are a professional Resume Analysis Assistant.

Answer the recruiter's question using ONLY the
provided resume information.

Instructions:
1. Do not invent candidate details.
2. Mention the resume filename as the source.
3. If information is unavailable, clearly say so.
4. Keep the answer simple and understandable.
5. Distinguish between explicit evidence and uncertain information.

Resume Information:
{context}

Recruiter's Question:
{question}

Answer:
"""

    print("\nGenerating AI answer...")

    try:

        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print("\n========== AI GENERATED ANSWER ==========")

        print(response["message"]["content"])

    except Exception as error:

        print("\nOllama connection error!")

        print("Reason:", error)

        print(
            "Make sure Ollama is installed, "
            "running, and qwen2.5:3b is downloaded."
        )


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("\nSMART RESUME INTELLIGENCE SYSTEM")

    print("----------------------------------")

    # Store or update resume information
    store_resumes()

    print("\nRAG SYSTEM READY!")

    print("Type 'search' to retrieve resume passages.")

    print("Type 'exit' to close the system.")

    while True:

        question = input(
            "\nAsk a question about resumes: "
        ).strip()

        if question.lower() == "exit":
            print("RAG system closed.")
            break

        if not question:
            print("Please enter a question.")
            continue

        if question.lower().startswith("search "):

            search_question = question[7:].strip()

            if search_question:
                search_resumes(search_question)
            else:
                print("Please enter a search question.")

        else:

            generate_answer(question)
