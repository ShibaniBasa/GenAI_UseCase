import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from query_filter import extract_filters
from memory import memory

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

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embedding_model
    )

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512
    )

    _initialized = True


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


def get_answer(question, session_id="default"):

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

    chat_history = memory.get_history(session_id)

    history_text = ""

    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are a Programming Assistant.

Use:
1. Previous Conversation History
2. Retrieved Context

Rules:
- Use chat history when the user asks follow-up questions.
- If information is not available in the context, say:

I don't have enough information in the provided documents.

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    memory.add_message(
        session_id,
        "User",
        question
    )

    memory.add_message(
        session_id,
        "Assistant",
        answer
    )

    return answer


def run_console_app():

    initialize_rag()

    session_id = "console_user"

    while True:

        question = input(
            "\nAsk Question (type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("\nApplication Closed.")
            break

        if not question:
            continue

        answer = get_answer(
            question,
            session_id
        )

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    run_console_app()