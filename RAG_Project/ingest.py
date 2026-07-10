import os
import shutil

from dotenv import load_dotenv

from utils import load_all_documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


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


vector_store = FAISS.from_documents(

    chunks,

    embedding_model

)

vector_store.save_local("vector_store")

print("Vector Store Created.")