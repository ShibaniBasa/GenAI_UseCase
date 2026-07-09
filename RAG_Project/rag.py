
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------

load_dotenv()

# -------------------------------------------------
# Load Embedding Model
# -------------------------------------------------

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# Load Vector Store
# -------------------------------------------------

print("Loading Vector Store...")

vector_store = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("Vector Store Loaded Successfully.")

# -------------------------------------------------
# Load LLM
# -------------------------------------------------

print("Loading Groq LLM...")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("RAG Ready!")

print("-"*80)

while True:

    question = input("\nAsk a Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    # ---------------------------------------------
    # Retrieve Similar Documents
    # ---------------------------------------------

    documents = vector_store.similarity_search(
        question,
        k=1
    )

    # ---------------------------------------------
    # Build Context
    # ---------------------------------------------

    context = ""

    for doc in documents:

        context += doc.page_content
        context += "\n\n"

    # ---------------------------------------------
    # Prompt
    # ---------------------------------------------

    prompt = f"""
You are a helpful programming tutor.

Answer ONLY using the provided context.

If the answer is not available in the context,
say "I don't have enough information."

Context:

{context}

Question:

{question}

Answer:
"""

    # ---------------------------------------------
    # Generate Answer
    # ---------------------------------------------

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    print("\n")
    print("="*80)
    print("ANSWER")
    print("="*80)

    print(response.content)

    print("\n")
    print("="*80)
    print("Retrieved Documents")
    print("="*80)

    for i, doc in enumerate(documents, start=1):

        print(f"\nDocument {i}")

        print("-"*50)

        print(doc.page_content)

        print("\nMetadata")

        for key, value in doc.metadata.items():
            print(f"{key} : {value}")

        print("-"*50)