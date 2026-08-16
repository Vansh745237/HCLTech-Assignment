"""Ingest PDF documents into one persistent ChromaDB collection."""
from pathlib import Path
from typing import Iterable

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "meridian_supply_chain"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # free, local, lightweight (no torch needed)
BATCH_SIZE = 100

_model = None


def get_embedding_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def load_pdfs(pdf_files: Iterable[Path] | None = None) -> list[dict]:
    files = sorted(pdf_files or DATA_DIR.glob("*.pdf"))
    if not files:
        raise FileNotFoundError("No PDF files found inside the data folder.")

    documents = []
    for pdf_file in files:
        reader = PdfReader(str(pdf_file))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append({"text": text, "file": pdf_file.name, "page": page_number})
    return documents


def create_chunks(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = []
    for document in documents:
        pieces = splitter.split_text(document["text"])
        for piece_index, text in enumerate(pieces):
            if text.strip():
                chunks.append({
                    "text": text.strip(),
                    "file": document["file"],
                    "page": document["page"],
                    "chunk_index": piece_index,
                })
    return chunks


def embed_texts(model: TextEmbedding, texts: list[str]) -> list[list[float]]:
    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vectors = list(model.embed(batch))
        embeddings.extend(v.tolist() for v in vectors)
    return embeddings


def store_in_chroma(chunks: list[dict], model: TextEmbedding | None = None) -> int:
    if not chunks:
        raise ValueError("No text chunks were created from the PDFs.")
    model = model or get_embedding_model()
    embeddings = embed_texts(model, [c["text"] for c in chunks])

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )

    if collection.count() > 0:
        collection.delete(where={"_source": "ingestion"})

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "file": c["file"],
            "page": int(c["page"]),
            "chunk_index": int(c["chunk_index"]),
            "_source": "ingestion",
        }
        for c in chunks
    ]
    collection.add(
        ids=ids,
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def ingest(pdf_paths: Iterable[Path] | None = None) -> dict:
    documents = load_pdfs(pdf_paths)
    chunks = create_chunks(documents)
    count = store_in_chroma(chunks)
    return {
        "files": len(set(d["file"] for d in documents)),
        "pages": len(documents),
        "chunks": count,
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
    }


def main() -> dict:
    result = ingest()
    print("===================================")
    print("Meridian Supply Chain RAG")
    print("===================================")
    print(f"Files processed : {result['files']}")
    print(f"Pages processed : {result['pages']}")
    print(f"Chunks stored   : {result['chunks']}")
    print(f"Collection      : {result['collection']}")
    print(f"Embeddings      : {result['embedding_model']}")
    return result


if __name__ == "__main__":
    main()
