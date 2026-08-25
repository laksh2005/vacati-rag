# Vacati RAG

A small retrieval-augmented generation service over Vacati's travel documentation — property pages, cancellation policies, visa notes, a tours catalogue and an in-residence menu. Ask a question over HTTP, get an answer that is grounded in those documents, with the sources it used. When the documents don't cover the question, it says so instead of guessing.

Everything runs on one Gemini API key: `gemini-embedding-001` for embeddings, `gemini-3.5-flash` for reranking and answering.

```
question ──► embed ──┬──► dense (cosine) ──┐
                     │                     ├──► RRF ──► LLM rerank ──► grounded? ──► answer + citations
                     └──► BM25 ────────────┘                              │
                                                                          └──► insufficient_context
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your Gemini key in it
python ingest.py              # builds index/ (~30s, costs well under a cent)
uvicorn app.main:app --reload
```

- `http://localhost:8000/` — demo page: ask questions, see the citations and the per-query cost
- `http://localhost:8000/docs` — interactive OpenAPI reference

```bash
pytest          # unit tests, Gemini mocked, no key or network needed
python eval.py  # retrieval quality numbers over the golden question set
```

> Free-tier Gemini keys are rate limited per minute *and* per day, per model. Every chat call retries with exponential backoff on a 429, so a single query still succeeds (it just pauses), but `eval.py` makes ~36 calls and can exhaust a free daily quota. Set `CHAT_MODEL` in `.env` to switch models, and keep the price table in `app/config.py` in step with whatever you pick.

## API

Base URL `http://localhost:8000`. Every `/v1` route needs an `X-API-Key` header; accepted keys come from `API_KEYS` in `.env`.

### `POST /v1/query`

```json
{ "query": "What is the cancellation policy for Villa Aurora?", "top_k": 5 }
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | string | yes | 3–1000 characters |
| `top_k` | int | no | 1–10, default 5. How many chunks ground the answer. |

Response:

```json
{
  "answer": "Cancellations more than 30 days before check-in are fully refunded...",
  "answer_type": "grounded",
  "citations": [
    {
      "chunk_id": "9f2c1ab4e77d0c31",
      "document": "cancellation-policy.md",
      "title": "Cancellation and Refund Policy",
      "section": "Cancellation and Refund Policy > Villa Aurora",
      "snippet": "Cancellations made more than 30 days before check-in receive a full refund...",
      "score": 0.95
    }
  ],
  "usage": { "input_tokens": 2614, "output_tokens": 304, "estimated_cost_usd": 0.001543, "cached": false },
  "latency_ms": 2518
}
```

**`answer_type` is the field to branch on, not the status code.** It is `grounded` when the answer is backed by at least one real citation, and `insufficient_context` when retrieval found nothing relevant enough or the model reported it could not answer from the context. Both come back as HTTP 200 with the same schema; `insufficient_context` responses carry an empty `citations` array, because nothing cleared the grounding bar.

### `GET /v1/documents`

Lists indexed documents and their chunk counts. Useful for confirming what the service can actually answer about.

### `GET /health`

No auth. Returns index size and the model ids in use.

### Errors

Every error has the same shape:

```json
{ "error": { "code": "unauthorized", "message": "Missing or invalid X-API-Key header." } }
```

| Status | `code` | When |
|---|---|---|
| 401 | `unauthorized` | Missing or unknown `X-API-Key` |
| 422 | `invalid_request` | Body failed validation (message names the field) |
| 429 | `rate_limited` | Per-key limit exceeded; a `Retry-After` header says when to try again |
| 502 | `upstream_error` | The Gemini call failed |

### Integrating

```bash
curl -s localhost:8000/v1/query \
  -H "Content-Type: application/json" -H "X-API-Key: demo-key" \
  -d '{"query":"Which dishes on the menu are vegan?"}'
```

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/query",
    headers={"X-API-Key": "demo-key"},
    json={"query": "Which dishes on the menu are vegan?"},
    timeout=30,
)
data = response.raise_for_status().json()

if data["answer_type"] == "grounded":
    print(data["answer"])
    for citation in data["citations"]:
        print("-", citation["section"], citation["document"])
else:
    print("Not covered by the documentation.")
```

## Chunking strategy

Two content types, two chunkers ([`app/chunker.py`](app/chunker.py)).

**Prose documents split on markdown headings.** In this corpus a `##` section *is* a unit of meaning: one policy, one property's rates, one FAQ answer. Splitting on a fixed window would cut Villa Aurora's refund windows in half and glue the tail onto Cliff House's; splitting on headings never does. Sections longer than ~450 tokens are split further at paragraph boundaries with ~60 tokens of overlap, and sections shorter than ~60 tokens are merged into the following one (the merged chunk keeps both headings, so citations stay honest).

Every chunk is embedded with its heading breadcrumb prefixed:

```
Cancellation and Refund Policy > Villa Aurora
Cancellations made more than 30 days before check-in receive a full refund...
```

That breadcrumb is doing real work. The corpus deliberately contains two near-identical cancellation policies — Villa Aurora (30/14 days) and Cliff House (14 days/48 hours). Without the breadcrumb their embeddings sit almost on top of each other and the model happily quotes the wrong one. With it, "Villa Aurora cancellation policy" lands on the right chunk.

**Catalogue records are one chunk each and are never split.** A tour or a menu item is an atomic fact; splitting it would separate a price from its dish. Each record is rendered into a flat sentence (`[MN-204] name: Grilled Sea Bream... price eur: 34. allergens: fish.`) rather than embedded as raw JSON, because the rendered form sits in the same space as the questions people actually ask. The structured fields are kept on the chunk as metadata.

Chunk ids are content hashes, so re-running `ingest.py` re-embeds only what changed.

## Retrieval

1. **Embed the query** with `task_type="RETRIEVAL_QUERY"`; documents were embedded as `RETRIEVAL_DOCUMENT`. Getting these the wrong way round costs quality silently — the two are different projections of the same model.
2. **Dense search** — cosine over a normalised matrix, top 20. Handles paraphrase: "can I bring my dog" → "one dog under 15 kg is permitted".
3. **BM25** — top 20. Handles the things embeddings blur: `TR-104`, `EUR 890`, "48 hours", "Cliff House".
4. **Reciprocal Rank Fusion** (k=60) to combine them. RRF works on ranks, so there is no need to normalise two incomparable score scales — a chunk that both methods like wins.
5. **Rerank** — one `gemini-3.5-flash` call scores all 15 fused candidates in a single request and returns JSON. One call, not fifteen. The reranker is explicitly told that a passage about a *different* property or product scores below 0.3, which is what keeps the two lookalike policies apart at the final step.

## Hallucination guardrails

Four layers, because none of them is sufficient alone:

1. **Threshold before generation.** If the best reranked chunk scores below `min_rerank_score` (0.35), the answer model is never called. Out-of-scope questions are cheap.
2. **Constrained prompt.** The system prompt allows only the numbered context blocks, and forbids answering a property-specific question from a block about a different property.
3. **Structured output.** `response_schema` forces `{answer, citations, sufficient_context}`, so the model reports insufficiency as a boolean instead of hedging in prose that a caller can't parse.
4. **Citation validation.** Citation numbers pointing at blocks we never sent are dropped. If nothing valid remains, the response is downgraded to `insufficient_context` — an answer with no surviving source is not a grounded answer, whatever it says about itself.

## Cost and latency

Measured end to end against a local server on this corpus (`gemini-3.5-flash`):

| Query | Calls | Latency | Tokens | Cost |
|---|---|---|---|---|
| Grounded answer | embed + rerank + answer | 2.5–3.9 s | ~2,600 in / ~300 out | ~$0.0015 |
| Out-of-scope (fallback) | embed + rerank | 2.5 s | ~1,800 in / ~250 out | ~$0.0012 |
| Repeat of any query | none | **0 ms** | — | **$0** |

Retrieval itself — dense, BM25 and RRF over 33 chunks — is under 5 ms and free; effectively all of the latency and all of the cost is the two model calls. That is what the choices below target:

- **Flash, not Pro**, with `thinking_budget=0`. Grounded extraction from five short passages does not need a reasoning model; it needs a fast one that follows instructions.
- **One rerank call**, not one per candidate. Scoring 15 passages in a single request is roughly 15× cheaper and ~10× faster than a per-passage cross-encoder loop.
- **768-dimension embeddings** instead of the 3072 default: a quarter of the index size and memory, with no measurable recall loss on this corpus. Vectors are re-normalised after truncation, which the model requires for dimensions other than 3072.
- **Response cache** keyed on the normalised query — a repeated question returns in 0 ms at $0 (verified above). FAQ-style traffic repeats heavily, so this is the single cheapest win available.
- **Skipping generation on low-relevance queries**, which removes the answer call from every out-of-scope question.
- Every response reports its own token usage and estimated cost, so a caller can meter spend without a separate billing integration. Prices live in [`app/config.py`](app/config.py).

## Retrieval quality

`python eval.py` scores 18 golden questions (14 answerable, 4 that the corpus deliberately cannot answer) and compares three pipelines:

| Strategy | Recall@5 |
|---|---|
| Dense only | _run `python eval.py`_ |
| Hybrid (dense + BM25) | |
| Hybrid + rerank | |

It also reports how often the guardrail makes the right answer/refuse call — the metric that matters most, since a confident wrong answer about a refund window is worse than no answer.

## Tradeoffs

- **A numpy array is the vector store.** For 33 chunks, brute-force cosine takes microseconds; an ANN index would add a dependency, a build step and approximation error to buy nothing. The swap point is in the Scaling section below.
- **The reranker is an LLM, not a cross-encoder.** A dedicated reranker would be faster and cheaper per call, but it is another provider and another key. One Gemini key for the whole system was worth ~1.2 s.
- **Cache and rate limiter are in-process dicts.** Correct for one instance, wrong the moment there are two. Redis is the fix and it is a small one; building it now would have been building for an audience that doesn't exist yet.
- **Token counts for embeddings are estimated** (~4 chars/token) since the embeddings endpoint doesn't return usage. Chat token counts are exact, from `usage_metadata`. Embeddings are a rounding error in the total either way.
- **The corpus is synthetic.** It was written to include hard cases — two near-identical policies, exact prices, allergen lists — rather than scraped from real properties.

## Scaling to a larger corpus

What breaks first, and what replaces it:

| Scale | What breaks | Fix |
|---|---|---|
| ~10k chunks | Index no longer trivially fits per-process; ingest is slow | Persist to a real store; parallelise embedding |
| ~100k chunks | Brute-force cosine crosses ~50 ms | HNSW (`hnswlib`, or the vector DB's own index) |
| ~1M chunks | In-process BM25 and in-memory vectors both stop fitting | Qdrant or pgvector with metadata pre-filtering; OpenSearch for lexical |
| Multi-instance | Cache and rate limiter diverge per process | Redis for both |
| Multi-tenant | Cross-tenant leakage risk | Tenant id as a mandatory filter on every retrieval, enforced at the store |

Two things about the current code make those swaps small rather than structural: retrieval is behind `retrieve()` in one file, and chunk ids are content hashes, so incremental re-indexing is a diff rather than a rebuild. Beyond that, the ingest script becomes a queue consumer triggered by document updates, and reranking moves to a dedicated reranker model once query volume makes the per-call LLM cost the dominant line item.

What would *not* change: the chunking strategy, the hybrid + rerank shape, and the guardrails. Those are corpus-shaped decisions, not scale-shaped ones.

## Layout

```
data/docs/*.md         prose corpus (properties, policies, guide, visa, FAQ)
data/catalog/*.json    structured corpus (tours, menu)
ingest.py              chunk -> embed -> index/
eval.py                retrieval quality + guardrail scoring
app/config.py          settings, model ids, price table
app/schemas.py         request/response contract
app/chunker.py         the two chunkers
app/gemini.py          every Gemini call: embed, rerank, answer
app/retriever.py       dense + BM25 + RRF + rerank
app/main.py            routes, auth, rate limit, cache, error handling
app/static/index.html  demo page
tests/test_rag.py      chunking, fusion, guardrails, auth, cache
```
