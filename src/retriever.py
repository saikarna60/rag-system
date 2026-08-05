"""Hybrid retrieval: dense (embeddings) + lexical (BM25), fused and reranked.

This mirrors a production RAG retriever:
  1. Dense search over sentence-transformer embeddings.
  2. Lexical BM25 search for exact keyword matches.
  3. Score fusion (weighted) into a single candidate ranking.
  4. Cross-encoder-style reranking of the top candidates.
"""
import math
import re
from collections import Counter

import numpy as np

from config import (
    EMBEDDING_MODEL, TOP_K, CANDIDATE_K, BM25_WEIGHT, DENSE_WEIGHT,
)


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Compact BM25 implementation for lexical scoring."""

    def __init__(self, corpus_tokens, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.N, 1)
        self.df = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in self.df.items()
        }

    def score(self, query_tokens, index: int):
        doc = self.corpus[index]
        freqs = Counter(doc)
        dl = len(doc)
        score = 0.0
        for t in query_tokens:
            if t not in freqs:
                continue
            idf = self.idf.get(t, 0.0)
            tf = freqs[t]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score


def _minmax(scores):
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        self._embedder = None
        self._embeddings = None
        self.bm25 = BM25([_tokenize(t) for t in self.texts])

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def build(self):
        """Precompute dense embeddings for the corpus."""
        self._embeddings = self.embedder.encode(
            self.texts, normalize_embeddings=True, show_progress_bar=False
        )
        return self

    def _dense_scores(self, query):
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        return (self._embeddings @ q).tolist()  # cosine (normalized)

    def _bm25_scores(self, query):
        qt = _tokenize(query)
        return [self.bm25.score(qt, i) for i in range(len(self.texts))]

    def retrieve(self, query: str, top_k: int = TOP_K):
        dense = _minmax(self._dense_scores(query))
        lexical = _minmax(self._bm25_scores(query))
        fused = [DENSE_WEIGHT * d + BM25_WEIGHT * l for d, l in zip(dense, lexical)]

        # Take candidates, then rerank.
        order = np.argsort(fused)[::-1][:CANDIDATE_K]
        reranked = self._rerank(query, order)
        return [
            {"chunk": self.chunks[i], "score": round(float(fused[i]), 4)}
            for i in reranked[:top_k]
        ]

    def _rerank(self, query, candidate_idx):
        """Lightweight cross-encoder-style rerank via token overlap + dense agreement.

        A production system swaps this for a Cohere Rerank or a cross-encoder model;
        the interface is identical.
        """
        qt = set(_tokenize(query))
        scored = []
        for i in candidate_idx:
            overlap = len(qt & set(_tokenize(self.texts[i]))) / (len(qt) + 1e-9)
            scored.append((i, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored]
