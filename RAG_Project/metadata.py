import json
import os

# ----------------------------------------------------
# Load Metadata Configuration
# ----------------------------------------------------

CONFIG_PATH = os.path.join(
    "config",
    "metadata.json"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)


# ----------------------------------------------------
# Get Metadata for a Document
# ----------------------------------------------------

def get_metadata(filename):

    metadata = METADATA.get(filename)

    if metadata is None:

        metadata = {
            "language": "Unknown",
            "document_type": "Unknown",
            "level": "Unknown",
            "format": "Unknown"
        }

    metadata["source_file"] = filename

    return metadata