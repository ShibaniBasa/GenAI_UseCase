import os

from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

from metadata import get_metadata


converter = DocumentConverter()


def create_document(file_path):

    filename = os.path.basename(file_path)

    print(f"Reading {filename}")

    result = converter.convert(file_path) #it returns a python object which we can access using . operator

    text = result.document.export_to_markdown()
    

    # -----------------------------
    # Load metadata from metadata.json
    # -----------------------------
    metadata = get_metadata(filename)

    return Document(
        page_content=text,
        metadata=metadata
    )


def load_all_documents(folder):

    documents = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        if os.path.isfile(path):

            try:

                document = create_document(path)

                documents.append(document)

            except Exception as e:

                print(f"Skipping {file}")
                print(e)

    return documents