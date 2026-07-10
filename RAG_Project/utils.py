import os

from langchain_core.documents import Document

from docling.document_converter import DocumentConverter


converter = DocumentConverter()


def create_document(file_path):

    filename = os.path.basename(file_path)

    print(f"Reading {filename}")

    result = converter.convert(file_path)

    text = result.document.export_to_markdown()

    return Document(
        page_content=text,
        metadata={
            "source_file": filename
        }
    )


def load_all_documents(folder):

    documents = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        if os.path.isfile(path):

            try:

                documents.append(create_document(path))

            except Exception as e:

                print(f"Skipping {file}")

                print(e)

    return documents