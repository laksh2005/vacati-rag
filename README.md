# Vacati RAG

A small RAG API over Vacati's travel docs: property pages, cancellation policies, visa notes, a tours catalogue and a menu. Ask a question, get an answer grounded in the docs with sources cited. If the docs don't cover it, it says so instead of guessing.

```
question -> embed -> dense search + BM25 -> RRF -> LLM rerank -> grounded? -> answer + citations
                                                                     |
                                                                     no -> insufficient_context
```

Runs on one Gemini key: `gemini-embedding-001` for embeddings, `gemini-3.5-flash` for reranking and answering.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Gemini key
python ingest.py              # builds index/
uvicorn app.main:app --reload
```

- `http://localhost:8000/` demo page
- `http://localhost:8000/docs` API reference

```bash
pytest                          # unit tests, no key needed
python eval.py --retrieval-only # dense vs hybrid recall, free, no model calls
python eval.py                  # full eval incl. rerank + guardrail, uses chat quota
```

Free-tier Gemini keys are capped per day per model. Every chat call retries on 429, but a full eval run needs ~40 calls, so use `--retrieval-only` if quota is tight. Change `CHAT_MODEL` in `.env` if needed.

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

**`GET /v1/documents`** lists indexed docs and chunk counts.
**`GET /health`** no auth, index size and model ids.

Errors are one shape: `{ "error": { "code": "...", "message": "..." } }`. 401 unauthorized, 422 invalid_request, 429 rate_limited, 502 upstream_error.

```bash
curl -s localhost:8000/v1/query \
  -H "Content-Type: application/json" -H "X-API-Key: demo-key" \
  -d '{"query":"Which dishes on the menu are vegan?"}'
```

## Chunking

**Prose splits on markdown headings.** A `##` section is a unit of meaning: one policy, one FAQ answer. Long sections split further with overlap; short ones merge into the next. Each chunk is embedded with its heading breadcrumb prefixed (`Cancellation Policy > Villa Aurora`), which is what keeps two near-identical property policies from getting confused.

**Catalogue records are one chunk each, never split.** A tour or menu item is one fact; splitting it separates a price from its name. Each record is rendered into a short sentence, not raw JSON, so it embeds close to how people actually ask.

## Retrieval

Query embedded as `RETRIEVAL_QUERY` (docs as `RETRIEVAL_DOCUMENT`, matters for quality). Dense cosine search catches paraphrase, BM25 catches exact strings (prices, property names). Both fused with Reciprocal Rank Fusion, then reranked in one Gemini call, which also produces the relevance score the guardrail thresholds on.

## Guardrails against hallucination

1. Below-threshold relevance skips generation entirely.
2. System prompt only allows the numbered context blocks, no outside knowledge.
3. Structured output forces `sufficient_context: bool` instead of hedging in prose.
4. Citations pointing at blocks never sent are dropped; if none remain, the answer downgrades to the fallback.

## Cost and latency

~$0.0015 and 2.5-4s per grounded query, almost entirely the two chat calls (rerank + answer), retrieval itself is under 5ms. Flash with thinking off, one batched rerank call instead of one per candidate, 768-dim embeddings, and an in-memory response cache that returns repeats at 0ms for $0.

## Retrieval quality

`python eval.py --retrieval-only` on 21 questions (17 answerable):

| Strategy | Recall@1 | Recall@5 |
|---|---|---|
| Dense only | 88% | 100% |
| Hybrid (dense + BM25) | 94% | 100% |

Recall@5 saturates fast on a 33-chunk corpus, recall@1 is what actually separates strategies, and only does so once the question set includes exact-string and cross-property cases. `python eval.py` fills in the rerank and guardrail rows on a chat-capable key.

## Tradeoffs

- Vector store is a numpy array on disk. Fine at this scale, swap point noted below.
- Reranker is an LLM call, not a dedicated cross-encoder, to keep everything on one key.
- Cache and rate limiter are in-process dicts: correct for one instance, not many.
- Corpus is synthetic, written to include hard cases on purpose.

## Scaling

| Scale | Breaks | Fix |
|---|---|---|
| ~10k chunks | Slow ingest | Persist properly, parallelise embedding |
| ~100k chunks | Brute-force cosine | HNSW or the vector DB's own index |
| ~1M chunks | In-memory BM25/vectors | Qdrant/pgvector + OpenSearch |
| Multi-instance | Cache/rate limiter diverge | Redis |
| Multi-tenant | Cross-tenant leakage | Mandatory tenant filter on retrieval |

Retrieval lives behind one `retrieve()` function and chunk ids are content hashes, so these swaps are localized, not structural. Chunking, the hybrid+rerank shape, and the guardrails wouldn't change; they're corpus-shaped decisions, not scale-shaped ones.

## Layout

```
data/docs/*.md, data/catalog/*.json   corpus
ingest.py                             chunk -> embed -> index/
eval.py                               retrieval + guardrail scoring
app/config.py, schemas.py             settings, request/response contract
app/chunker.py                        the two chunkers
app/gemini.py                         every Gemini call: embed, rerank, answer
app/retriever.py                      dense + BM25 + RRF + rerank
app/main.py                           routes, auth, rate limit, cache
app/static/index.html                 demo page
tests/test_rag.py                     chunking, fusion, guardrails, auth, cache
```
