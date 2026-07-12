import os
import shutil

from dotenv import load_dotenv

from utils import load_all_documents

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()


# ---------------------------------------------------
# Load Documents
# ---------------------------------------------------

print("Loading documents...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(BASE_DIR, "data")

print("DATA PATH:", DATA_FOLDER)
print("DATA EXISTS:", os.path.exists(DATA_FOLDER))

documents = load_all_documents(DATA_FOLDER)

print(f"{len(documents)} documents loaded.")


# ---------------------------------------------------
# Initialize Embedding Model
# ---------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------
# Semantic Chunking
# ---------------------------------------------------

print("Creating semantic chunks...")

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)

# Split Document objects while preserving metadata
chunks = semantic_splitter.split_documents(documents)
print("Chunks created:", len(chunks))
for chunk in chunks[:3]:   
    print(chunk.metadata)
print(f"{len(chunks)} semantic chunks created.")


# ---------------------------------------------------
# Remove Existing Vector Store
# ---------------------------------------------------

persist_directory = "vector_store"

if os.path.exists(persist_directory):
    print("Removing existing vector store...")
    shutil.rmtree(persist_directory)


# ---------------------------------------------------
# Create Chroma Vector Database
# ---------------------------------------------------

print("Creating Chroma Vector Store...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Vector Store Created Successfully!")
 
print(f"Total Documents : {len(documents)}")
print(f"Total Chunks    : {len(chunks)}")