import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from query_filter import extract_filters


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

embedding_model = None
vector_store = None
llm = None
_initialized = False


def initialize_rag():
    global embedding_model, vector_store, llm, _initialized

    if _initialized:
        return

    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            "Vector store not found. Please run ingest.py before starting rag.py."
        )

    print("Loading Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading Vector Store...")
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embedding_model
    )

    print("Vector Store Loaded Successfully.")
    print("Total Chunks in Chroma:", vector_store._collection.count())

    print("Loading LLM...")
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512
    )

    _initialized = True
    print("RAG Ready!")
    print("-" * 80)


def build_context(retrieved_docs):
    context = ""
    source_files = set()

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source_file", "Unknown")
        source_files.add(source)

        context += f"""

Document {i}

Source File:
{source}

Language:
{doc.metadata.get("language", "Unknown")}

Document Type:
{doc.metadata.get("document_type", "Unknown")}

Level:
{doc.metadata.get("level", "Unknown")}

Format:
{doc.metadata.get("format", "Unknown")}

Content:

{doc.page_content}

"""

    return context, sorted(source_files)


def get_answer(question):
    initialize_rag()

    question = (question or "").strip()

    if not question:
        return "Please provide a question."

    filters = extract_filters(question)

    if filters:
        retrieved_docs = vector_store.similarity_search(
            query=question,
            k=2,
            filter=filters
        )
    else:
        retrieved_docs = vector_store.similarity_search(
            query=question,
            k=2
        )

    if not retrieved_docs:
        return "I don't have enough information in the provided documents."

    context, _ = build_context(retrieved_docs)

    prompt = f"""
You are a Programming Assistant.

Answer ONLY using the provided context.

Rules:

1. Never use outside knowledge.

2. Answer only from the provided context.

3. If the answer is not available in the context, reply exactly:

I don't have enough information in the provided documents.

Context:

{context}

Question:

{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def run_console_app():
    initialize_rag()

    while True:
        question = input("\nAsk Question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("\nApplication Closed.")
            break

        if not question:
            continue

        print("\nAnswer:\n")
        print(get_answer(question))


if __name__ == "__main__":
    run_console_app()
