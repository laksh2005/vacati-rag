# Vacati RAG

A small RAG API over Vacati's travel docs: property pages, cancellation policies, visa notes, a tours catalogue and a menu. Ask a question, get an answer grounded in the docs with sources cited. If the docs don't cover it, it says so instead of guessing.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Gemini key
python ingest.py              # builds index/
uvicorn app.main:app --reload
```

This starts a server, it does not open a file. Once it is running, open **`http://localhost:8000/`** in your browser for the demo page (do not open `app/static/index.html` directly, it needs the server running), or **`http://localhost:8000/docs`** for the API reference.

```bash
pytest                          # unit tests, no key needed
python eval.py --retrieval-only # checks search quality, free, no model calls
python eval.py                  # full check, uses your Gemini quota
```

Free Gemini keys have a daily limit per model. Every call retries if it hits that limit, but a full `eval.py` run uses around 40 calls, so use `--retrieval-only` if you are low on quota.

## API

Every `/v1` request needs an `X-API-Key` header.

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

Check `answer_type`, not the HTTP status. `grounded` means the answer has at least one real source. `insufficient_context` means the documents did not have a good answer, either because nothing relevant was found or the model could not answer from what it was given. Both cases return a normal 200 response.

`GET /v1/documents` lists the indexed documents. `GET /health` does not need a key.

Every error looks the same: `{ "error": { "code": "...", "message": "..." } }`. 401 means a missing or wrong key, 422 means bad input, 429 means too many requests, 502 means the model call failed.

```bash
curl -s localhost:8000/v1/query \
  -H "Content-Type: application/json" -H "X-API-Key: demo-key" \
  -d '{"query":"Which dishes on the menu are vegan?"}'
```

## Design decisions

**Chunking.** Plain text documents are split by heading. Each `##` section is one policy or one answer, so splitting there keeps the meaning whole. Long sections get split again with a bit of overlap, and short ones get joined with the next. Catalog items like tours and menu entries are kept as one chunk each, splitting them would separate a price from what it belongs to. Every chunk also gets its heading path added before it is turned into a vector, for example `Cancellation Policy > Villa Aurora`, so two similar-looking policies for different properties don't get mixed up.

**Retrieval.** The question is turned into a vector the same way the documents were, then two kinds of search run at once: one that matches meaning (good for different wording of the same question) and one that matches exact words like prices or property names. Their results are combined, then a Gemini call scores how relevant each result actually is, this score is also what decides whether to answer at all.

**Stopping hallucinations.** Four checks, since no single one is enough on its own:
1. If nothing scores high enough, the model is never even asked, the answer is skipped straight to "not enough information."
2. The model is told to only use the given passages, nothing else.
3. The model has to explicitly say whether it had enough information, instead of just guessing in normal text.
4. Any source the model claims that wasn't actually given to it gets removed. If none are left, the answer is replaced with "not enough information."

**Cost and speed.** A normal answer costs about $0.0015 and takes 2.5 to 4 seconds, almost all of that time is the two model calls (scoring results, then writing the answer). Search itself is under 5 milliseconds. To keep costs down: a fast, lighter model, one scoring call instead of one per result, smaller vectors, and repeated questions are cached and cost nothing.

**How well retrieval works.** Run with `python eval.py --retrieval-only` on 21 test questions (17 have real answers):

| Method | Top answer correct | Correct answer in top 5 |
|---|---|---|
| Meaning search only | 88% | 100% |
| Meaning + exact word search | 94% | 100% |

With this few documents, almost everything shows up somewhere in the top 5, so that number alone doesn't tell you much. Whether the top answer is right is the more useful number, and combining both search methods does better there. `python eval.py` (the full version) also checks how well the model decides when to answer versus say "I don't know."

## Tradeoffs

- The document vectors are stored in a plain file on disk. That's fine for this size, searching them takes microseconds. With far more documents, this would need a proper vector database, and that change would only need to happen in one place in the code.
- The result-scoring step uses a general Gemini call instead of a model built just for that. This keeps everything on one API key, but a dedicated model would likely be faster and cheaper.
- Caching and rate limits are stored in memory. That works for one server, but would need to move to something shared, like Redis, if there were more than one.
- The documents are made up for this project, written on purpose to include tricky cases, like two similar policies for different properties.

## Scaling

Roughly what would need to change as the number of documents grows. At around 10,000 chunks, loading everything in gets slow, so that step would run in parallel instead of one at a time. At around 100,000 chunks, searching the plain file gets slow too, so that would move to a proper search index or vector database. Past that, even keyword search needs a dedicated tool instead of running in memory. If the app runs on more than one server, the cache and rate limits need to move somewhere shared instead of living inside each server. None of this changes how documents are split, how search works, or the safety checks, those depend on the content, not the size.
