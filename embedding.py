import json
import re
import uuid

import tiktoken
from constants import CHUNK_OVERLAP, CHUNK_SIZE
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import logging
from typing import Dict, Any

# Initialize logger for this module
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def preprocess_pdf_content(content: str) -> str:
    """Preprocess the content for embedding"""
    content = re.sub(r"\s+", " ", content)
    content = content.strip()
    content = re.sub(r"[^\w\s.,!?-]", "", content)
    return content


def num_tokens_from_string(string: str) -> int:
    """Approximate number of tokens in a string"""
    return len(string.split())


def format_for_embedding(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a chunk for embedding processing.

    Args:
        chunk_data: Dictionary containing chunk content and metadata
    Returns:
        Dictionary formatted for embedding
    """
    return {
        "text": chunk_data["content"],
        "metadata": {
            **chunk_data["metadata"],
            "token_count": chunk_data["token_count"],
            "chunk_id": chunk_data.get("chunk_id"),
        },
    }


def load_pdf_files(file_path: str):
    """Load and process PDF files"""
    logger.info(f"Loading PDF file: {file_path}")

    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Split documents into chunks
    docs = text_splitter.split_documents(documents)

    # Process each chunk
    processed_docs = []
    for doc in docs:
        # Preprocess the content
        content = preprocess_pdf_content(doc.page_content)

        # Update the document with processed content
        doc.page_content = content
        doc.metadata.update(
            {
                "chunk_id": str(uuid.uuid4()),
                "token_count": num_tokens_from_string(content),
                "source": file_path,
            }
        )
        processed_docs.append(doc)
    logger.info(f"Processed {len(processed_docs)} chunks from {file_path}")
    return processed_docs
