"""Hybrid retrieval: dense + BM25, fused with RRF, then reranked by the LLM."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.gemini import embed_texts, rerank

CHUNKS_FILE = "chunks.json"
VECTORS_FILE = "vectors.npy"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Index:
    chunks: list[dict]
    vectors: np.ndarray      # (n_chunks, embed_dim), L2-normalised
    bm25: BM25Okapi

    @property
    def size(self) -> int:
        return len(self.chunks)


def load_index(index_dir: str | None = None) -> Index:
    path = Path(index_dir or settings.index_dir)
    chunks = json.loads((path / CHUNKS_FILE).read_text(encoding="utf-8"))
    vectors = np.load(path / VECTORS_FILE)
    bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks])
    return Index(chunks=chunks, vectors=vectors, bm25=bm25)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _rrf(*ranked_lists: list[int]) -> list[int]:
    """Reciprocal Rank Fusion: combine rankings without comparing their scores."""
    fused: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def embed_query(query: str) -> np.ndarray:
    return normalize(np.array([embed_texts([query], "RETRIEVAL_QUERY")[0]]))[0]


def dense_ranking(index: Index, query_vector: np.ndarray, k: int) -> list[int]:
    """Cosine similarity. Catches paraphrase: 'can I bring my dog' -> 'pets are permitted'."""
    return np.argsort(index.vectors @ query_vector)[::-1][:k].tolist()


def lexical_ranking(index: Index, query: str, k: int) -> list[int]:
    """BM25. Catches exact strings: prices, property names, 'TR-104', '48 hours'."""
    return np.argsort(index.bm25.get_scores(tokenize(query)))[::-1][:k].tolist()


def candidates(index: Index, query: str) -> list[int]:
    """Both rankings, fused with RRF."""
    query_vector = embed_query(query)
    return _rrf(
        dense_ranking(index, query_vector, settings.dense_k),
        lexical_ranking(index, query, settings.bm25_k),
    )


def retrieve(index: Index, query: str, top_k: int) -> tuple[list[dict], int, int]:
    """Full pipeline. Returns chunks with a `score`, best first, plus token usage."""
    fused = candidates(index, query)[: settings.rerank_k]
    shortlist = [index.chunks[i] for i in fused]

    scores, in_tokens, out_tokens = rerank(query, shortlist)
    ranked = sorted(
        ({**chunk, "score": scores.get(i, 0.0)} for i, chunk in enumerate(shortlist)),
        key=lambda c: c["score"],
        reverse=True,
    )
    return ranked[:top_k], in_tokens, out_tokens
