import json
import os
import re

CONFIG_PATH = os.path.join(
    "config",
    "metadata.json"
)

with open(CONFIG_PATH, "r") as f:
    METADATA = json.load(f)


def extract_filters(question):

    question = question.lower()

    words = re.findall(r"[a-zA-Z0-9\+#-]+", question)

    filters = {}

    languages = set()
    document_types = set()
    levels = set()
    formats = set()

    for file_metadata in METADATA.values():

        languages.add(file_metadata["language"].lower())
        document_types.add(file_metadata["document_type"].lower())
        levels.add(file_metadata["level"].lower())
        formats.add(file_metadata["format"].lower())

    for language in languages:

        if language == "c":
            if " c " in f" {question} ":
                filters["language"] = "C"

        elif language == "cpp":
            if "cpp" in words or "c++" in question:
                filters["language"] = "CPP"

        elif language in words:
            filters["language"] = language.capitalize()

    for doc_type in document_types:

        if doc_type in words:
            filters["document_type"] = doc_type.capitalize()

    for level in levels:

        level_words = level.lower().split("-")

        if all(word in words for word in level_words):
            filters["level"] = level

    for file_format in formats:

        if file_format in words:
            filters["format"] = file_format.upper()

    return filters