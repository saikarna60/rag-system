"""Agentic RAG: a multi-step workflow that decides how to answer.

This models the LangGraph pattern used in production agentic-RAG systems:

    ┌─ route ──> needs_retrieval? ──yes──> retrieve ─> grade ─> generate
    │                                                    │
    └────────────────── no ──> answer_directly          └─(low relevance)─> rewrite_query ─> retrieve

Here the graph is implemented explicitly (no framework dependency) so the control
flow is visible and testable. In the repo's LangGraph variant, each node below maps
to a graph node with the same responsibility.
"""
from rag import RAGPipeline


class AgenticRAG:
    def __init__(self):
        self.rag = RAGPipeline().build()
        self.max_rewrites = 1

    # --- Graph nodes ------------------------------------------------------
    def route(self, question: str) -> str:
        """Decide whether the question needs retrieval or can be answered directly."""
        chit_chat = {"hi", "hello", "thanks", "thank you", "bye"}
        if question.lower().strip().strip("!?.") in chit_chat:
            return "direct"
        return "retrieve"

    def grade(self, hits) -> bool:
        """Grade retrieved context relevance; low scores trigger a query rewrite."""
        if not hits:
            return False
        return hits[0]["score"] >= 0.15

    def rewrite(self, question: str) -> str:
        """Expand the query to improve recall on a weak first pass."""
        return f"{question} definition explanation details"

    # --- Orchestration ----------------------------------------------------
    def run(self, question: str) -> dict:
        trace = []
        route = self.route(question)
        trace.append(f"route → {route}")

        if route == "direct":
            trace.append("answer_directly")
            return {"answer": "Hi! Ask me anything about the knowledge base.",
                    "sources": [], "trace": trace}

        q = question
        for attempt in range(self.max_rewrites + 1):
            hits = self.rag.retriever.retrieve(q)
            trace.append(f"retrieve (attempt {attempt + 1}) → {len(hits)} hits")
            if self.grade(hits):
                trace.append("grade → relevant")
                break
            trace.append("grade → weak, rewriting query")
            q = self.rewrite(question)

        result = self.rag.answer(question)
        result["trace"] = trace
        return result


if __name__ == "__main__":
    agent = AgenticRAG()
    for q in ["hello", "What is reranking in RAG?"]:
        print(f"\nQ: {q}")
        out = agent.run(q)
        print("Trace:", " | ".join(out["trace"]))
        print("Answer:", out["answer"][:160])
