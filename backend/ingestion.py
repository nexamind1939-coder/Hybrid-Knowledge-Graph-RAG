
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS


STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/app/storage"))
FAISS_DIR = STORAGE_DIR / "faiss"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)


def get_embeddings():
    """Create the local FastEmbed embedding model."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_pdf(file_path: str, original_name: str):
    """Extract, chunk, embed, and save one PDF into the FAISS index."""

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No readable text found in {original_name}")

    # 2. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(f"No text chunks created from {original_name}")

    # 3. Add source and page information
    for chunk in chunks:
        chunk.metadata["source"] = original_name

        # PyPDFLoader uses zero-based page numbers.
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    # 4. Generate embeddings
    embeddings = get_embeddings()

    # 5. Create or update the FAISS index
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    index_file = FAISS_DIR / "index.faiss"

    if index_file.exists():
        vector_store = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        new_vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.merge_from(new_vector_store)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    # 6. Persist the index
    vector_store.save_local(str(FAISS_DIR))

    return {
        "file_name": original_name,
        "pages_processed": len(documents),
        "chunks_created": len(chunks),
        "status": "success"
    }
