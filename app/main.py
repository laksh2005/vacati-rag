"""FastAPI app: auth, rate limiting, caching, and the routes.

The whole HTTP surface is in this file so it can be read top to bottom.
"""

import hashlib
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("vacati_rag")

from app.chunker import est_tokens
from app.config import settings
from app.gemini import GeminiError, generate_answer
from app.retriever import Index, load_index, retrieve
from app.schemas import (
    Citation,
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Usage,
)

FALLBACK_ANSWER = (
    "I don't have enough information in the indexed documents to answer that. "
    "Try rephrasing, or ask about our properties, policies, tours or menu."
)

state: dict[str, Index] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The index is small enough to hold in memory; load it once at startup.
    state["index"] = load_index()
    yield
    state.clear()


app = FastAPI(
    title="Vacati RAG API",
    version="1.0.0",
    description=(
        "Retrieval-augmented Q&A over Vacati's travel documentation.\n\n"
        "Send `X-API-Key` with every `/v1` request. Answers are grounded in the indexed "
        "documents and always carry citations; when retrieval finds nothing relevant, the "
        "response comes back with `answer_type` set to `insufficient_context` and no "
        "citations rather than a guess. Branch on `answer_type`, not on the status code."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Auth and rate limiting: in-memory, single process. See README > Scaling.
# --------------------------------------------------------------------------
_requests: dict[str, deque[float]] = defaultdict(deque)


def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    if x_api_key not in settings.allowed_keys:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    window = _requests[x_api_key]
    now = time.time()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit of {settings.rate_limit_per_minute} requests/minute exceeded.",
            headers={"Retry-After": str(int(60 - (now - window[0])) + 1)},
        )
    window.append(now)
    return x_api_key


# --------------------------------------------------------------------------
# Response cache: a repeated question costs nothing and returns instantly.
# Keyed per API key too, so one caller never gets served another caller's
# cached answer (and its usage/cost numbers) for the "same" question.
# --------------------------------------------------------------------------
_cache: dict[str, tuple[float, QueryResponse]] = {}
CACHE_MAX_ENTRIES = 1000


def cache_key(api_key: str, query: str, top_k: int) -> str:
    return hashlib.sha256(f"{api_key}|{query.strip().lower()}|{top_k}".encode()).hexdigest()


def cache_get(key: str) -> QueryResponse | None:
    hit = _cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    _cache.pop(key, None)
    return None


def cache_set(key: str, response: QueryResponse) -> None:
    if len(_cache) >= CACHE_MAX_ENTRIES:
        # Cheap unbounded-growth guard: drop the oldest-expiring entries first.
        oldest = sorted(_cache, key=lambda k: _cache[k][0])[: len(_cache) - CACHE_MAX_ENTRIES + 1]
        for stale in oldest:
            _cache.pop(stale, None)
    _cache[key] = (time.time() + settings.cache_ttl_seconds, response)


def estimate_cost(input_tokens: int, output_tokens: int, embed_tokens: int) -> float:
    return round(
        input_tokens / 1e6 * settings.price_chat_input
        + output_tokens / 1e6 * settings.price_chat_output
        + embed_tokens / 1e6 * settings.price_embed_input,
        6,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.post(
    "/v1/query",
    response_model=QueryResponse,
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    summary="Ask a question, get a grounded answer with sources",
)
def query(body: QueryRequest, api_key: str = Depends(require_api_key)) -> QueryResponse:
    started = time.perf_counter()

    key = cache_key(api_key, body.query, body.top_k)
    cached = cache_get(key)
    if cached:
        return cached.model_copy(
            update={
                "usage": cached.usage.model_copy(
                    update={"cached": True, "estimated_cost_usd": 0.0}
                ),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )

    chunks, rerank_in, rerank_out = retrieve(state["index"], body.query, body.top_k)
    embed_tokens = est_tokens(body.query)

    # Guardrail 1: nothing cleared the relevance bar, so skip the answer model entirely.
    if not chunks or chunks[0]["score"] < settings.min_rerank_score:
        return _respond(
            FALLBACK_ANSWER, "insufficient_context", [],
            rerank_in, rerank_out, embed_tokens, started, key,
        )

    answer, answer_in, answer_out = generate_answer(body.query, chunks)
    input_tokens = rerank_in + answer_in
    output_tokens = rerank_out + answer_out

    # Guardrail 2: drop citations pointing at blocks we never sent, then require at
    # least one real citation. A "grounded" answer with no source is not grounded.
    valid = [n for n in answer.citations if 1 <= n <= len(chunks)]
    if not answer.sufficient_context or not valid:
        return _respond(
            FALLBACK_ANSWER, "insufficient_context", [],
            input_tokens, output_tokens, embed_tokens, started, key,
        )

    citations = [
        Citation(
            chunk_id=chunks[n - 1]["id"],
            document=chunks[n - 1]["doc"],
            title=chunks[n - 1]["title"],
            section=chunks[n - 1]["section"],
            snippet=chunks[n - 1]["text"][:300],
            score=round(chunks[n - 1]["score"], 3),
        )
        for n in dict.fromkeys(valid)
    ]
    return _respond(
        answer.answer, "grounded", citations,
        input_tokens, output_tokens, embed_tokens, started, key,
    )


def _respond(
    answer: str,
    answer_type: str,
    citations: list[Citation],
    in_tokens: int,
    out_tokens: int,
    embed_tokens: int,
    started: float,
    key: str,
) -> QueryResponse:
    response = QueryResponse(
        answer=answer,
        answer_type=answer_type,
        citations=citations,
        usage=Usage(
            input_tokens=in_tokens + embed_tokens,
            output_tokens=out_tokens,
            estimated_cost_usd=estimate_cost(in_tokens, out_tokens, embed_tokens),
        ),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    cache_set(key, response)
    return response


@app.get("/v1/documents", response_model=list[DocumentInfo], summary="List indexed documents")
def documents(_key: str = Depends(require_api_key)) -> list[DocumentInfo]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for chunk in state["index"].chunks:
        counts[(chunk["doc"], chunk["title"])] += 1
    return [
        DocumentInfo(document=doc, title=title, chunks=n)
        for (doc, title), n in sorted(counts.items())
    ]


@app.get("/health", response_model=HealthResponse, summary="Liveness and index size (no auth)")
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks_indexed=state["index"].size if "index" in state else 0,
        embed_model=settings.embed_model,
        chat_model=settings.chat_model,
    )


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# --------------------------------------------------------------------------
# One error shape for the whole API: {"error": {"code", "message"}}
# --------------------------------------------------------------------------
def error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    codes = {401: "unauthorized", 404: "not_found", 429: "rate_limited"}
    return error(exc.status_code, codes.get(exc.status_code, "error"), str(exc.detail), exc.headers)


@app.exception_handler(RequestValidationError)
def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"][1:])
    return error(422, "invalid_request", f"{field}: {first['msg']}")


@app.exception_handler(GeminiError)
def gemini_error(request: Request, exc: GeminiError) -> JSONResponse:
    # Log the raw provider error server-side; never hand it to the caller,
    # it can carry internal details we don't want to expose over the API.
    logger.error("Gemini call failed: %s", exc)
    return error(502, "upstream_error", "The language model call failed. Try again shortly.")
