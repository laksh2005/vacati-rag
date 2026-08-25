"""Measure the retrieval stack against eval_questions.yaml.

Usage:  python eval.py

Prints recall@5 for three strategies (dense only, hybrid, hybrid + rerank) so the
choice of pipeline is backed by numbers, plus how often the guardrail correctly
refuses the questions the corpus cannot answer.
"""

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


def hits(chunks: list[dict], case: dict) -> bool:
    """Did any of the top chunks come from the expected document and section?"""
    return any(
        c["doc"] == case["doc"] and case["section"].lower() in c["section"].lower()
        for c in chunks[:TOP_K]
    )


def refuses(query: str, chunks: list[dict]) -> bool:
    """Run the same two guardrails the API applies and report whether they fire."""
    if not chunks or chunks[0]["score"] < settings.min_rerank_score:
        return True
    answer, _, _ = generate_answer(query, chunks)
    valid = [n for n in answer.citations if 1 <= n <= len(chunks)]
    return not answer.sufficient_context or not valid


def main() -> None:
    index = load_index()
    cases = yaml.safe_load(open("eval_questions.yaml", encoding="utf-8"))
    answerable = [c for c in cases if c["answerable"]]
    print(f"{len(cases)} questions ({len(answerable)} answerable)\n")

    scores = {"dense only": 0, "hybrid (dense + BM25)": 0, "hybrid + rerank": 0}
    guardrail_correct = 0

    for number, case in enumerate(cases, start=1):
        query = case["question"]
        vector = embed_query(query)
        reranked, _, _ = retrieve(index, query, TOP_K)

        if case["answerable"]:
            dense = [index.chunks[i] for i in dense_ranking(index, vector, TOP_K)]
            fused = [
                index.chunks[i]
                for i in _rrf(
                    dense_ranking(index, vector, settings.dense_k),
                    lexical_ranking(index, query, settings.bm25_k),
                )
            ]
            scores["dense only"] += hits(dense, case)
            scores["hybrid (dense + BM25)"] += hits(fused, case)
            scores["hybrid + rerank"] += hits(reranked, case)

        refused = refuses(query, reranked)
        guardrail_correct += refused == (not case["answerable"])
        print(f"  [{number}/{len(cases)}] {'refused' if refused else 'answered':<8} {query[:60]}")

    print("\nRecall@5")
    for name, hit_count in scores.items():
        print(f"  {name:<24} {hit_count / len(answerable):>4.0%}  ({hit_count}/{len(answerable)})")

    print("\nGuardrail")
    print(f"  answer / refuse decision {guardrail_correct / len(cases):>4.0%}  ({guardrail_correct}/{len(cases)})")


if __name__ == "__main__":
    main()
