"""FastAPI service exposing the RAG pipeline.

Run:  uvicorn app:app --reload   (from src/)
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag import RAGPipeline

app = FastAPI(title="RAG Document Q&A", version="1.0.0")

# Build the pipeline once at startup.
pipeline = RAGPipeline().build()


class Query(BaseModel):
    question: str = Field(..., example="What is hybrid retrieval?")
    top_k: int = Field(4, ge=1, le=10)


class Source(BaseModel):
    source: str
    score: float
    preview: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    used_llm: bool


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=Answer)
def ask(query: Query):
    return pipeline.answer(query.question, top_k=query.top_k)
