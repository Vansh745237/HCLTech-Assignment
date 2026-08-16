"""Retrieval + Groq (Llama) answering for the Meridian supply-chain RAG system."""
import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "meridian_supply_chain"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, local, must match ingest.py
LLM_MODEL = "llama-3.3-70b-versatile"  # free via Groq
DEFAULT_TOP_K = 6

REFUSAL = "The information is not available in the uploaded documents."


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is missing. Add it to .env.")
    return Groq(api_key=key)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Chroma collection does not exist. Upload/index the PDFs first."
        ) from exc


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    question = question.strip()
    if not question:
        return []
    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    embedding = model.encode([question], show_progress_bar=False)[0].tolist()

    n = min(max(int(top_k), 1), collection.count())
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    chunks = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        chunks.append({
            "text": document,
            "file": metadata.get("file", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "distance": distance,
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"SOURCE {i}\nDocument: {c['file']}\nPage: {c['page']}\nContent:\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


SYSTEM_PROMPT = f"""You are an internal Supply Chain Assistant for Meridian Components Pvt. Ltd.

Answer ONLY from the supplied context. Do not use outside knowledge and do not invent or guess.
If the context does not contain enough information to answer the question, reply exactly:
{REFUSAL}

For questions requiring multiple documents, combine facts only when they are supported by the supplied context.
Be precise with names, dates, quantities, percentages, currency amounts, policy clauses, and actions.
When a policy clause applies, state the clause number and the required buyer/supplier action if present in the context.
Do not calculate a value unless the required inputs and formula are explicitly present in the context.
"""


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return {"answer": REFUSAL, "sources": [], "chunks": []}

    context = build_context(chunks)
    response = get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
    )
    answer = (response.choices[0].message.content or REFUSAL).strip()

    sources = []
    seen = set()
    for chunk in chunks:
        key = (chunk["file"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": chunk["file"], "page": chunk["page"]})
    return {"answer": answer, "sources": sources, "chunks": chunks}


if __name__ == "__main__":
    question = input("Enter your finance/supply-chain question: ")
    result = answer_question(question)
    print("\nANSWER\n" + result["answer"])
    print("\nSOURCES")
    for source in result["sources"]:
        print(f"- {source['file']} | Page {source['page']}")
