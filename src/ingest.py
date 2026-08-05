"""Document loading and chunking.

Splits raw documents into overlapping character windows, preserving source
metadata so retrieved chunks can be cited back to their origin.
"""
from dataclasses import dataclass, field
from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, source: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Sliding-window chunker with overlap to preserve context across boundaries."""
    text = " ".join(text.split())  # normalize whitespace
    chunks, start, cid = [], 0, 0
    while start < len(text):
        end = start + size
        piece = text[start:end]
        # Try to break on a sentence boundary near the end for cleaner chunks.
        if end < len(text):
            last_period = piece.rfind(". ")
            if last_period > size * 0.5:
                piece = piece[: last_period + 1]
                end = start + last_period + 1
        chunks.append(Chunk(text=piece.strip(), source=source, chunk_id=cid))
        cid += 1
        start = end - overlap
    return chunks


def load_documents(data_dir: Path = DATA_DIR):
    """Load every .txt / .md file under data_dir into chunks. Seeds a sample if empty."""
    files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))
    if not files:
        _seed_sample_docs(data_dir)
        files = list(data_dir.glob("*.txt"))

    all_chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(chunk_text(text, source=f.name))
    return all_chunks


def _seed_sample_docs(data_dir: Path):
    """Write a couple of sample knowledge-base documents so the demo runs out of the box."""
    (data_dir / "ml_basics.txt").write_text(
        "Retrieval-augmented generation (RAG) combines a retriever and a generator. "
        "The retriever finds relevant passages from a knowledge base using semantic "
        "search over vector embeddings, and the generator (a large language model) "
        "produces an answer grounded in those passages. RAG reduces hallucination "
        "because the model is conditioned on retrieved evidence rather than relying "
        "only on its parameters. Hybrid retrieval blends lexical search (BM25) with "
        "dense vector search to capture both exact keyword matches and semantic "
        "similarity. Rerankers, often cross-encoders, reorder candidate passages by "
        "relevance before they are sent to the language model.",
        encoding="utf-8",
    )
    (data_dir / "vector_search.txt").write_text(
        "Vector databases such as FAISS, Chroma, Pinecone, and Qdrant store dense "
        "embeddings and support approximate nearest-neighbor search. Embeddings are "
        "produced by models like sentence-transformers, which map text into a "
        "high-dimensional space where semantically similar texts are close together. "
        "Chunking splits long documents into smaller overlapping windows so that "
        "retrieval returns focused, relevant context. Metadata filtering narrows the "
        "search to a subset of documents, for example by date or author.",
        encoding="utf-8",
    )
