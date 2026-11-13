
import chromadb
import ollama

from sentence_transformers import SentenceTransformer
from pathlib import Path


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_DIR = BASE_DIR / "data" / "processed_text"
VECTOR_DIR = BASE_DIR / "models" / "resume_vector_db"

RESUME_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD EMBEDDING MODEL
# =========================================================

print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# 3. CONNECT TO CHROMADB
# =========================================================

client = chromadb.PersistentClient(
    path=str(VECTOR_DIR)
)

collection = client.get_or_create_collection(
    name="resume_knowledge",
    metadata={"hnsw:space": "cosine"}
)

print("ChromaDB connected successfully!")


# =========================================================
# 4. TEXT CHUNKING
# =========================================================

def create_chunks(text, chunk_size=1000, overlap=150):

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


# =========================================================
# 5. STORE RESUMES IN CHROMADB
# =========================================================

def store_resumes():

    resume_files = list(RESUME_DIR.glob("*.txt"))

    if not resume_files:
        print("No processed resumes found!")
        return 0

    total_chunks = 0

    for resume_file in resume_files:

        print(f"\nProcessing: {resume_file.name}")

        try:

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

        except Exception as error:

            print(
                f"Error processing {resume_file.name}: {error}"
            )

    print("\n========== RAG DATABASE REPORT ==========")

    print("Total chunks processed:", total_chunks)

    print("Total database records:", collection.count())

    return total_chunks


# =========================================================
# 6. RETRIEVE RELEVANT RESUME INFORMATION
# =========================================================

def retrieve_context(question, chat_history=None, top_k=5):
    if chat_history is None:
        chat_history = []
    # Keep your existing RAG code below this line.
    if collection.count() == 0:
        return []

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    context = []

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents or not documents[0]:
        return []

    for i, document in enumerate(documents[0]):

        metadata = metadatas[0][i] or {}

        distance = (
            distances[0][i]
            if distances and distances[0]
            else None
        )

        context.append({
            "source": metadata.get("source", "Unknown"),
            "content": document,
            "distance": distance
        })

    return context


# =========================================================
# 7. SEMANTIC SEARCH
# =========================================================

def search_resumes(question, top_k=5):

    retrieved_data = retrieve_context(
        question,
        top_k=top_k
    )

    if not retrieved_data:
        print("No resume information found.")
        return []

    print("\n========== SEARCH RESULTS ==========")

    for i, item in enumerate(retrieved_data, start=1):

        distance = item["distance"]

        similarity = (
            max(0, (1 - distance) * 100)
            if distance is not None
            else 0
        )

        print(f"\nResult {i}")

        print("Resume:", item["source"])

        print("Similarity:", round(similarity, 2), "%")

        print("Content:", item["content"][:500])

    return retrieved_data


# =========================================================
# 8. GENERATE ANSWERS USING OLLAMA
# =========================================================

def generate_answer(question, chat_history=None, top_k=5):

    if not question or not question.strip():
        return "Please enter a question."

    try:

        # Retrieve relevant resume information
        retrieved_data = retrieve_context(
            question,
            top_k=top_k
        )

        # Prepare resume context
        context = ""

        for item in retrieved_data:

            context += (
                f"\nResume Source: {item['source']}\n"
                f"Resume Content: {item['content']}\n"
            )

        if not context:
            context = "No relevant resume information found."

        # Prepare conversation history
        history = ""

        if chat_history:

            for message in chat_history[-10:]:

                role = message.get("role", "user")
                content = message.get("content", "")

                if role in ["user", "assistant"] and content:
                    history += f"\n{role}: {content}\n"

        if not history:
            history = "No previous conversation."

        # Create prompt
        prompt = f"""
You are an intelligent AI Recruiter Assistant
and a helpful technical learning assistant.

YOUR RESPONSIBILITIES:

1. Answer different questions asked by the user.
2. Explain NLP, Machine Learning, Python, SQL,
   Deep Learning, RAG and Generative AI.
3. Answer resume-related questions using retrieved
   candidate information.
4. Explain candidate skills, education and projects.
5. Compare candidates when requested.
6. Explain technical concepts with examples.
7. Use previous conversation when relevant.
8. Give a fresh, relevant answer to every new question.
9. Avoid repeating the same answer unnecessarily.
10. Use simple and understandable English.

IMPORTANT RULES:

- For general technical questions, use your general
  knowledge. Do not force irrelevant resume context
  into the answer.
- For candidate-related questions, use the supplied
  resume information.
- Never invent candidate qualifications or experience.
- If candidate information is unavailable, say so.
- Mention the resume filename when using candidate data.
- If the user asks a follow-up question, understand
  it using the previous conversation.
- Answer the current question directly.

RESUME INFORMATION:

{context}

PREVIOUS CONVERSATION:

{history}

CURRENT QUESTION:

{question}

Provide a clear and useful answer:
"""

        print("\nGenerating AI answer...")

        # Call Ollama
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.4
            }
        )

        answer = response["message"]["content"].strip()

        if not answer:
            return "The AI model returned an empty answer."

        print("\n========== AI GENERATED ANSWER ==========")

        print(answer)

        # Return answer to Streamlit
        return answer

    except Exception as error:

        error_message = (
            f"AI answer generation failed: {str(error)}"
        )

        print(error_message)

        return error_message


# =========================================================
# 9. MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("\nSMART RESUME INTELLIGENCE SYSTEM")

    print("----------------------------------")

    # Store resumes in ChromaDB
    store_resumes()

    print("\nRAG SYSTEM READY!")

    print("Ask any question about resumes or technical topics.")

    print("Type 'exit' to close the system.")

    while True:

        question = input(
            "\nAsk a question: "
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

            answer = generate_answer(question)

            print("\nAI Answer:")
            print(answer)
