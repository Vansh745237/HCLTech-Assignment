"""Optional FastAPI backend for the Meridian Supply Chain RAG system."""
from pathlib import Path
import shutil
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ingest import ingest
from rag import DEFAULT_TOP_K, EMBEDDING_MODEL, LLM_MODEL, get_collection, answer_question

app = FastAPI(
    title="Meridian Supply Chain RAG API",
    version="1.0.0",
    description="Ingest Meridian supply-chain PDFs and ask grounded questions.",
)

DATA_DIR = Path("data")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=4, le=10)


@app.get("/stats")
def stats():
    try:
        collection = get_collection()
        total = collection.count()
    except Exception:
        total = 0
    return {
        "collection": "meridian_supply_chain",
        "total_chunks": total,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
    }


@app.post("/ingest")
async def ingest_files(files: Annotated[list[UploadFile], File(...)]):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")
    DATA_DIR.mkdir(exist_ok=True)
    paths = []
    try:
        for upload in files:
            if not upload.filename or not upload.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
            safe_name = Path(upload.filename).name
            path = DATA_DIR / safe_name
            with path.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
            paths.append(path)
        result = ingest(paths)
        get_collection.clear()
        return {"files": result["files"], "chunks": result["chunks"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: AskRequest):
    try:
        result = answer_question(request.question, top_k=request.top_k)
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
