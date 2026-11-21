
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ingestion import ingest_pdf


# Create FastAPI application FIRST
app = FastAPI(
    title="Hybrid Knowledge Graph RAG",
    description="PDF-based Hybrid RAG with FAISS and Knowledge Graph",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Storage directories
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/app/storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Home endpoint
@app.get("/")
async def home():
    return {
        "message": "Hybrid Knowledge Graph RAG API",
        "status": "running"
    }


# Health endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "FastAPI",
        "version": "1.0.0"
    }


# PDF upload endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    safe_name = Path(file.filename).name
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = UPLOAD_DIR / unique_name

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_pdf(str(file_path), safe_name)

        return {
            "message": "PDF uploaded successfully",
            "result": result
        }

    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        await file.close()
