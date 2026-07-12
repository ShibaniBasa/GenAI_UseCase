import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from query_filter import extract_filters


load_dotenv()

print("Loading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Vector Store...")

vector_store = Chroma(
    persist_directory="vector_store",
    embedding_function=embedding_model
)

print("Vector Store Loaded Successfully.")
print("Total Chunks in Chroma:", vector_store._collection.count())
print("Loading LLM...")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("RAG Ready!")
print("-" * 80)


while True:

    question = input(
        "\nAsk Question (type 'exit' to quit): "
    ).strip()

    if question.lower() == "exit":
        print("\nApplication Closed.")
        break

    filters = extract_filters(question)

    print("\nDetected Filters")

    if filters:

        for key, value in filters.items():
            print(f"{key}: {value}")

    else:

        print("No metadata filters detected.")

    print("\nSearching documents...")
    


    if filters:

        retrieved_docs = vector_store.similarity_search(
            query=question,
            k=5,
            filter=filters
        )

    else:

        retrieved_docs = vector_store.similarity_search(
            query=question,
            k=5
        )

    print(f"Retrieved {len(retrieved_docs)} document(s).")

    if not retrieved_docs:

        print("\nNo matching documents found.")
        continue

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

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(response.content)

    print("\n")
    print("=" * 80)
    print("SOURCE FILES")
    print("=" * 80)

    for source in sorted(source_files):
        print(f"- {source}")

    print("\n")
    print("=" * 80)
    print("DOCUMENT METADATA")
    print("=" * 80)

    for i, doc in enumerate(retrieved_docs, start=1):

        print(f"\nDocument {i}")

        for key, value in doc.metadata.items():
            print(f"{key}: {value}")