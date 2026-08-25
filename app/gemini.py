"""Every Gemini call in the project lives here: embed, rerank, answer.

Keeping them in one file means there is exactly one place to look for what we
send the model, what it costs, and how failures are surfaced.
"""

import time
from typing import Callable

from google import genai
from google.genai import errors, types

from app.config import settings
from app.schemas import GroundedAnswer, RerankScore

_client: genai.Client | None = None

EMBED_BATCH = 100
RETRIES = 3


class GeminiError(RuntimeError):
    """Any failure talking to Gemini. Surfaced to callers as HTTP 502."""


def client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise GeminiError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _call(what: str, request: Callable):
    """Run a Gemini call, backing off on 429s and wrapping everything else."""
    for attempt in range(RETRIES + 1):
        try:
            return request()
        except errors.ClientError as exc:
            if exc.code == 429 and attempt < RETRIES:
                time.sleep(settings.retry_base_seconds * 2**attempt)
                continue
            raise GeminiError(f"{what} failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - one boundary for all SDK errors
            raise GeminiError(f"{what} failed: {exc}") from exc
    raise GeminiError(f"{what} failed: still rate limited after {RETRIES} retries")


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a list of texts.

    task_type matters: documents are embedded as RETRIEVAL_DOCUMENT and queries
    as RETRIEVAL_QUERY. Mismatching them costs retrieval quality silently.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        response = _call(
            "embedding",
            lambda batch=batch: client().models.embed_content(
                model=settings.embed_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embed_dim,
                ),
            ),
        )
        vectors.extend(e.values for e in response.embeddings)
    return vectors


RERANK_SYSTEM = (
    "You score how well each numbered passage answers the user's question. "
    "Score 0 to 1: 1 means the passage directly contains the answer, 0 means it is unrelated. "
    "A passage about a different property, product or policy than the one asked about scores below 0.3. "
    "Return a score for every passage."
)


def rerank(query: str, candidates: list[dict]) -> tuple[dict[int, float], int, int]:
    """Score all candidates in ONE call. Returns {index: score}, input and output tokens."""
    passages = "\n\n".join(
        f"[{i}] {c['section']}\n{c['text']}" for i, c in enumerate(candidates)
    )
    response = _call(
        "rerank",
        lambda: client().models.generate_content(
            model=settings.chat_model,
            contents=f"Question: {query}\n\nPassages:\n{passages}",
            config=types.GenerateContentConfig(
                system_instruction=RERANK_SYSTEM,
                temperature=0,
                response_mime_type="application/json",
                response_schema=list[RerankScore],
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        ),
    )

    scores = {row.id: row.score for row in (response.parsed or [])}
    return scores, _in_tokens(response), _out_tokens(response)


ANSWER_SYSTEM = (
    "You answer questions about Vacati's travel documentation.\n"
    "Rules:\n"
    "1. Use ONLY the numbered context blocks provided. Never use outside knowledge.\n"
    "2. Cite the block numbers you used in the citations field.\n"
    "3. If the blocks do not contain the answer, set sufficient_context to false and "
    "leave the answer short. Do not guess, and do not fill gaps from general knowledge.\n"
    "4. If the question is about a specific property, product or date range, only answer "
    "from blocks about that exact one. A similar-looking block about something else is not an answer.\n"
    "5. Be concise and concrete. Quote exact figures, dates and prices from the blocks."
)


def generate_answer(query: str, chunks: list[dict]) -> tuple[GroundedAnswer, int, int]:
    """Answer strictly from the given chunks. Returns the answer, input and output tokens."""
    context = "\n\n".join(
        f"[{i + 1}] {c['section']}\n{c['text']}" for i, c in enumerate(chunks)
    )
    response = _call(
        "generation",
        lambda: client().models.generate_content(
            model=settings.chat_model,
            contents=f"Context blocks:\n{context}\n\nQuestion: {query}",
            config=types.GenerateContentConfig(
                system_instruction=ANSWER_SYSTEM,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=GroundedAnswer,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        ),
    )

    answer = response.parsed
    if answer is None:
        raise GeminiError("generation returned no parseable answer")
    return answer, _in_tokens(response), _out_tokens(response)


def _in_tokens(response) -> int:
    return getattr(response.usage_metadata, "prompt_token_count", 0) or 0


def _out_tokens(response) -> int:
    usage = response.usage_metadata
    return (getattr(usage, "candidates_token_count", 0) or 0) + (
        getattr(usage, "thoughts_token_count", 0) or 0
    )
