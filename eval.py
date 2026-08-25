"""Measure the retrieval stack against eval_questions.yaml.

Usage:  python eval.py [--retrieval-only]

Prints recall@5 for three strategies (dense only, hybrid, hybrid + rerank) so the
choice of pipeline is backed by numbers, plus how often the guardrail correctly
refuses the questions the corpus cannot answer.

--retrieval-only scores just the two strategies that need no model calls. It is
free and instant, which makes it the right loop when tuning chunking or fusion —
and the fallback when a free-tier daily quota is spent.
"""

import sys

import yaml

from app.config import settings
from app.gemini import generate_answer
from app.retriever import (
    Index,
    _rrf,
    dense_ranking,
    embed_query,
    lexical_ranking,
    load_index,
    retrieve,
)

TOP_K = 5


def hits(chunks: list[dict], case: dict, k: int = TOP_K) -> bool:
    """Did any of the top k chunks come from the expected document and section?"""
    return any(
        c["doc"] == case["doc"] and case["section"].lower() in c["section"].lower()
        for c in chunks[:k]
    )


def refuses(query: str, chunks: list[dict]) -> bool:
    """Run the same two guardrails the API applies and report whether they fire."""
    if not chunks or chunks[0]["score"] < settings.min_rerank_score:
        return True
    answer, _, _ = generate_answer(query, chunks)
    valid = [n for n in answer.citations if 1 <= n <= len(chunks)]
    return not answer.sufficient_context or not valid


def main(retrieval_only: bool = False) -> None:
    index = load_index()
    cases = yaml.safe_load(open("eval_questions.yaml", encoding="utf-8"))
    answerable = [c for c in cases if c["answerable"]]
    print(f"{len(cases)} questions ({len(answerable)} answerable)\n")

    # Recall@1 and @5: with a corpus this small @5 saturates, so @1 is the metric
    # that actually separates the strategies.
    scores = {name: [0, 0] for name in ("dense only", "hybrid (dense + BM25)", "hybrid + rerank")}
    guardrail_correct = 0

    for number, case in enumerate(cases, start=1):
        query = case["question"]
        vector = embed_query(query)
        reranked = [] if retrieval_only else retrieve(index, query, TOP_K)[0]

        if case["answerable"]:
            dense = [index.chunks[i] for i in dense_ranking(index, vector, TOP_K)]
            fused = [
                index.chunks[i]
                for i in _rrf(
                    dense_ranking(index, vector, settings.dense_k),
                    lexical_ranking(index, query, settings.bm25_k),
                )
            ]
            for name, ranked in (
                ("dense only", dense),
                ("hybrid (dense + BM25)", fused),
                ("hybrid + rerank", reranked),
            ):
                scores[name][0] += hits(ranked, case, k=1)
                scores[name][1] += hits(ranked, case, k=TOP_K)

        if retrieval_only:
            print(f"  [{number}/{len(cases)}] {query[:60]}")
            continue

        refused = refuses(query, reranked)
        guardrail_correct += refused == (not case["answerable"])
        print(f"  [{number}/{len(cases)}] {'refused' if refused else 'answered':<8} {query[:60]}")

    total = len(answerable)
    print(f"\n  {'':<22} {'recall@1':>8} {'recall@5':>9}")
    for name, (at_1, at_5) in scores.items():
        if retrieval_only and name == "hybrid + rerank":
            continue
        print(f"  {name:<22} {at_1 / total:>8.0%} {at_5 / total:>9.0%}")

    if not retrieval_only:
        print("\nGuardrail")
        print(f"  answer / refuse decision {guardrail_correct / len(cases):>4.0%}  ({guardrail_correct}/{len(cases)})")


if __name__ == "__main__":
    main(retrieval_only="--retrieval-only" in sys.argv)
