# Vacati RAG

A small RAG API over Vacati's travel docs: property pages, cancellation policies, visa notes, a tours catalogue and a menu. Ask a question, get an answer grounded in the docs with sources cited. If the docs don't cover it, it says so instead of guessing.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Gemini key
python ingest.py              # builds index/
uvicorn app.main:app --reload
```

This starts a server, it does not open a file. Once it's running, open **`http://localhost:8000/`** in a browser for the demo page (do not open `app/static/index.html` directly, it needs the API behind it), or **`http://localhost:8000/docs`** for the interactive API reference.

```bash
pytest                          # unit tests, no key needed
python eval.py --retrieval-only # dense vs hybrid recall, free, no model calls
python eval.py                  # full eval incl. rerank + guardrail, uses chat quota
```

Free-tier Gemini keys are capped per day per model. Every chat call retries on 429, but a full eval run needs ~40 calls, so use `--retrieval-only` if quota is tight.

## API

Every `/v1` route needs an `X-API-Key` header.

**`POST /v1/query`**

```json
{ "query": "What is the cancellation policy for Villa Aurora?", "top_k": 5 }
```

```json
{
  "answer": "Cancellations more than 30 days before check-in are fully refunded...",
  "answer_type": "grounded",
  "citations": [
    { "document": "cancellation-policy.md", "section": "Cancellation and Refund Policy > Villa Aurora", "snippet": "...", "score": 0.95 }
  ],
  "usage": { "input_tokens": 2740, "output_tokens": 355, "estimated_cost_usd": 0.0017, "cached": false },
  "latency_ms": 2500
}
```

`answer_type` is what to branch on, not the status code. `grounded` means at least one real citation backs the answer; `insufficient_context` means retrieval found nothing relevant, or the model couldn't answer from the context, either way returned as HTTP 200 with empty citations.

`GET /v1/documents` lists indexed docs. `GET /health` needs no auth.

Errors are one shape: `{ "error": { "code": "...", "message": "..." } }`. 401 unauthorized, 422 invalid_request, 429 rate_limited, 502 upstream_error.

```bash
curl -s localhost:8000/v1/query \
  -H "Content-Type: application/json" -H "X-API-Key: demo-key" \
  -d '{"query":"Which dishes on the menu are vegan?"}'
```

## Design decisions

**Chunking.** Prose splits on markdown headings, a `##` section is one policy or one FAQ answer, splitting mid-section would break the meaning. Long sections split further with overlap; short ones merge into the next. Catalogue records are one chunk each, never split, a tour or menu item is one fact, splitting it separates a price from its name. Every chunk is embedded with a heading breadcrumb prefixed (`Cancellation Policy > Villa Aurora`), which is what keeps two near-identical property policies from getting confused with each other.

**Retrieval.** Query embedded as `RETRIEVAL_QUERY`, docs as `RETRIEVAL_DOCUMENT` (mismatching these costs quality silently). Dense cosine search catches paraphrase, BM25 catches exact strings like prices and property names. Both are fused with Reciprocal Rank Fusion, then reranked in one batched Gemini call, which also produces the relevance score the guardrail thresholds on.

**Hallucination guardrails**, four layers, none of them sufficient alone:
1. Below-threshold relevance skips generation entirely.
2. System prompt only allows the numbered context blocks, no outside knowledge.
3. Structured output forces `sufficient_context: bool` instead of hedging in prose.
4. Citations pointing at blocks never sent are dropped; if none remain, the answer downgrades to the fallback.

**Cost and latency.** ~$0.0015 and 2.5-4s per grounded query, almost entirely the two chat calls (rerank + answer), retrieval itself is under 5ms. Flash with thinking off, one batched rerank call instead of one per candidate, 768-dim embeddings, and an in-memory response cache that returns repeats at 0ms for $0.

**Retrieval quality.** `python eval.py --retrieval-only` on 21 questions (17 answerable):

| Strategy | Recall@1 | Recall@5 |
|---|---|---|
| Dense only | 88% | 100% |
| Hybrid (dense + BM25) | 94% | 100% |

Recall@5 saturates fast on this small a corpus, recall@1 is what separates the strategies. `python eval.py` fills in the rerank and guardrail rows on a chat-capable key.

## Tradeoffs

- Vector store is a numpy array on disk. Correct at this scale (brute-force cosine takes microseconds over ~35 chunks); would become HNSW or a real vector DB past ~100k chunks, and the swap is localized since retrieval lives behind one `retrieve()` function.
- Reranker is an LLM call, not a dedicated cross-encoder, to keep everything on one Gemini key. A dedicated reranker would be faster and cheaper per call.
- Cache and rate limiter are in-process dicts: correct for one instance, would move to Redis for multiple.
- Corpus is synthetic, written to include hard cases (near-identical policies, exact prices) on purpose.

## Scaling

What breaks first as the corpus grows, and roughly where: past ~10k chunks, ingest gets slow, that's a queue and parallel embedding, not a rewrite. Past ~100k chunks, brute-force cosine stops being instant, that's HNSW or a managed vector DB (Qdrant, pgvector). Past that, in-memory BM25 stops fitting too, and lexical search moves to something like OpenSearch. Multiple instances means the cache and rate limiter need to move to Redis, they're in-process dicts today. None of this touches chunking, the hybrid+rerank shape, or the guardrails, those are decisions shaped by the content, not by scale.
