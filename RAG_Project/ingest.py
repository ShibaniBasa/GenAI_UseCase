from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_all_documents
import os
load_dotenv()
docs=load_all_documents('data')
chunks=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100).split_documents(docs)
emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.from_documents(chunks,emb)
os.makedirs('vector_store',exist_ok=True)
db.save_local('vector_store')
print('Done')
