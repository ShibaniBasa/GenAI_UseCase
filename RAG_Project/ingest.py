import os
import shutil                  #shutil stands for Shell Utilities, python built in modules high-level file and folder operations like copying, moving, and deleting.

from dotenv import load_dotenv

from utils import load_all_documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma


load_dotenv()


print("Loading documents...")

documents = load_all_documents("data")

print(f"{len(documents)} documents loaded.")


splitter = RecursiveCharacterTextSplitter(

    chunk_size=400,

    chunk_overlap=80,

    separators=[

        "\n# ",

        "\n## ",

        "\n### ",

        "\n\n",

        "\n",

        ". ",

        " "

    ]

)

chunks = splitter.split_documents(documents)

print(f"{len(chunks)} chunks created.")


embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)


if os.path.exists("vector_store"):

    shutil.rmtree("vector_store")

vector_store = Chroma.from_documents(

    documents=chunks,

    embedding=embedding_model,

    persist_directory="vector_store"

)

print("Vector Store Created.")