
import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()


# ----------------------------------------------------
# Load Embedding Model
# ----------------------------------------------------

print("Loading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ----------------------------------------------------
# Load FAISS Vector Store
# ----------------------------------------------------

print("Loading Vector Store...")

vector_store = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("Vector Store Loaded Successfully.")


# ----------------------------------------------------
# Load LLM
# ----------------------------------------------------

print("Loading LLM...")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("RAG Ready!")
print("-" * 80)


# ====================================================
# Chat Loop
# ====================================================

while True:

    question = input("\nAsk Question (type 'exit' to quit): ").strip()

    if question.lower() == "exit":
        break

    # ------------------------------------------------
    # Retrieve Relevant Chunks
    # ------------------------------------------------

    retrieved_docs = vector_store.similarity_search(
        question,
        k=2
    )

    if not retrieved_docs:
        print("\nNo relevant information found.")
        continue

    # ------------------------------------------------
    # Display Retrieved Chunks
    # ------------------------------------------------

    #print("\n")
    #print("=" * 80)
    #print("RETRIEVED CHUNKS")
    #print("=" * 80)

    context = ""

    source_files = set()

    for i, doc in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get("source_file", "Unknown")

        source_files.add(source)

        #print(f"\nChunk {i}")
        #print("-" * 50)
        #print(f"Source : {source}\n")
        #print(doc.page_content[:700])
        #print()

        context += f"""
Document {i}

Source File:
{source}

Content:
{doc.page_content}

"""

    # ------------------------------------------------
    # Prompt
    # ------------------------------------------------

    prompt = f"""
You are a helpful Programming Assistant.

Answer ONLY using the information provided in the context.

Rules:

1. Do NOT use outside knowledge.

2. Ignore any information that is not relevant to the user's question.

3. If multiple documents contain relevant information,
combine them into one clear answer.

4. If the answer is only partially available,
provide only the available information.

5. If the answer is NOT present in the context,
reply exactly with:

I don't have enough information in the provided documents.

6. Do not hallucinate.

Context:

{context}

Question:

{question}

Answer:
"""

    # ------------------------------------------------
    # Generate Response
    # ------------------------------------------------

    response = llm.invoke(prompt)

    # ------------------------------------------------
    # Print Answer
    # ------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(response.content)

    # ------------------------------------------------
    # Print Sources
    # ------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SOURCE FILES")
    print("=" * 80)

    for source in sorted(source_files):
        print(f"- {source}")