"""RAG pipeline: retrieve context, then generate a grounded answer.

If an OpenAI key is configured, the LLM writes the answer conditioned on the
retrieved context. Otherwise the system returns an extractive answer built from
the most relevant chunk — so the pipeline is fully runnable without any API key.
"""
from config import USE_LLM, LLM_MODEL, OPENAI_API_KEY, TOP_K
from ingest import load_documents
from retriever import HybridRetriever


SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the provided "
    "context. If the answer is not in the context, say you don't know. Cite the "
    "source filename for any claim."
)


class RAGPipeline:
    def __init__(self):
        self.retriever = None

    def build(self):
        chunks = load_documents()
        self.retriever = HybridRetriever(chunks).build()
        return self

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        hits = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            f"[{h['chunk'].source}] {h['chunk'].text}" for h in hits
        )
        sources = [
            {"source": h["chunk"].source, "score": h["score"],
             "preview": h["chunk"].text[:160] + "…"}
            for h in hits
        ]

        if USE_LLM:
            answer = self._generate_with_llm(question, context)
        else:
            answer = self._extractive_answer(question, hits)

        return {"answer": answer, "sources": sources, "used_llm": USE_LLM}

    def _generate_with_llm(self, question: str, context: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    def _extractive_answer(self, question: str, hits) -> str:
        """No-LLM fallback: return the most relevant passage as the answer."""
        if not hits:
            return "I couldn't find anything relevant in the knowledge base."
        top = hits[0]["chunk"]
        return (
            f"Based on {top.source}: {top.text}\n\n"
            f"(Set OPENAI_API_KEY to get a synthesized answer instead of the raw passage.)"
        )


if __name__ == "__main__":
    rag = RAGPipeline().build()
    for q in ["What is hybrid retrieval?", "Which vector databases are mentioned?"]:
        print(f"\nQ: {q}")
        result = rag.answer(q)
        print(f"A: {result['answer'][:300]}")
        print(f"Sources: {[s['source'] for s in result['sources']]}")
