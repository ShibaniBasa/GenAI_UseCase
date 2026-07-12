import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS
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

print("Loading LLM...")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("RAG Ready!")
print("-" * 80)


def filter_documents(documents, filters):

    if not filters:
        return documents

    filtered_docs = []

    for doc in documents:

        matched = True

        for key, value in filters.items():

            metadata_value = doc.metadata.get(key)

            if metadata_value is None:
                matched = False
                break

            if str(metadata_value).lower() != str(value).lower():
                matched = False
                break

        if matched:
            filtered_docs.append(doc)

    return filtered_docs


while True:

    question = input(
        "\nAsk Question (type 'exit' to quit): "
    ).strip()

    if question.lower() == "exit":
        break

    filters = extract_filters(question)

    print("\nDetected Filters")

    if filters:

        for key, value in filters.items():
            print(f"{key}: {value}")

    else:

        print("No metadata filters detected.")

    retrieved_docs = vector_store.similarity_search(
        question,
        k=20
    )

    retrieved_docs = filter_documents(
        retrieved_docs,
        filters
    )

    retrieved_docs = retrieved_docs[:2]

    if not retrieved_docs:

        print("\nNo matching documents found.")

        continue

    context = ""

    source_files = set()

    for i, doc in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get(
            "source_file",
            "Unknown"
        )

        source_files.add(source)

        language = doc.metadata.get(
            "language",
            "Unknown"
        )

        document_type = doc.metadata.get(
            "document_type",
            "Unknown"
        )

        level = doc.metadata.get(
            "level",
            "Unknown"
        )

        file_format = doc.metadata.get(
            "format",
            "Unknown"
        )

        context += f"""
        Document {i}

Source File:
{source}

Language:
{language}

Document Type:
{document_type}

Level:
{level}

Format:
{file_format}

Content:

{doc.page_content}

"""
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
   

print("\nApplication Closed.")