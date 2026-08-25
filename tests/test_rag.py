"""Tests run with no API key and no network: Gemini is stubbed out."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.chunker import chunk_markdown, chunk_records
from app.retriever import Index, _rrf, normalize
from app.schemas import GroundedAnswer

DATA = Path("data")


# --- chunking -------------------------------------------------------------
def test_markdown_splits_on_headings_and_keeps_the_breadcrumb():
    chunks = chunk_markdown(DATA / "docs" / "cancellation-policy.md")
    sections = [c["section"] for c in chunks]

    assert "Cancellation and Refund Policy > Villa Aurora" in sections
    assert "Cancellation and Refund Policy > Cliff House" in sections
    # The two near-identical policies must not end up in the same chunk.
    aurora = next(c for c in chunks if c["section"].endswith("Villa Aurora"))
    assert "30 days" in aurora["text"] and "48 hours" not in aurora["text"]
    # The breadcrumb is what we embed, so lookalike sections stay distinguishable.
    assert aurora["embed_text"].startswith("Cancellation and Refund Policy > Villa Aurora")


def test_records_are_one_chunk_each_and_never_split():
    chunks = chunk_records(DATA / "catalog" / "menu.json")
    assert len(chunks) == 8
    item = next(c for c in chunks if "MN-204" in c["section"])
    assert "34" in item["text"] and "sea bream" in item["text"].lower()


def test_chunk_ids_are_stable_content_hashes():
    first = chunk_markdown(DATA / "docs" / "villa-aurora.md")
    second = chunk_markdown(DATA / "docs" / "villa-aurora.md")
    assert [c["id"] for c in first] == [c["id"] for c in second]


# --- retrieval maths ------------------------------------------------------
def test_rrf_rewards_agreement_between_both_rankings():
    # 7 is second in both rankings; 1 and 9 are first in only one each.
    assert _rrf([1, 7], [9, 7])[0] == 7


def test_normalize_gives_unit_rows():
    rows = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


# --- API ------------------------------------------------------------------
CHUNK = {
    "id": "abc123",
    "doc": "cancellation-policy.md",
    "title": "Cancellation and Refund Policy",
    "section": "Cancellation and Refund Policy > Villa Aurora",
    "text": "Cancellations more than 30 days before check-in receive a full refund.",
    "embed_text": "...",
    "meta": {},
    "score": 0.9,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_index", lambda: Index(chunks=[CHUNK], vectors=np.zeros((1, 4)), bm25=None))
    monkeypatch.setattr(main, "retrieve", lambda index, query, top_k: ([CHUNK], 100, 10))
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda query, chunks: (GroundedAnswer(answer="Full refund over 30 days out.", citations=[1], sufficient_context=True), 200, 20),
    )
    main._cache.clear()
    main._requests.clear()
    with TestClient(main.app) as test_client:
        yield test_client


HEADERS = {"X-API-Key": "demo-key"}


def test_query_returns_a_grounded_answer_with_citations(client):
    body = client.post("/v1/query", json={"query": "Villa Aurora cancellation?"}, headers=HEADERS).json()

    assert body["answer_type"] == "grounded"
    assert body["citations"][0]["section"].endswith("Villa Aurora")
    assert body["usage"]["estimated_cost_usd"] > 0


def test_low_relevance_falls_back_without_calling_the_answer_model(client, monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda index, query, top_k: ([{**CHUNK, "score": 0.1}], 100, 10))
    monkeypatch.setattr(main, "generate_answer", _boom)

    body = client.post("/v1/query", json={"query": "what is the wifi password"}, headers=HEADERS).json()
    assert body["answer_type"] == "insufficient_context"
    assert body["citations"] == []


def test_invented_citations_are_dropped_and_downgrade_the_answer(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda query, chunks: (GroundedAnswer(answer="Made up.", citations=[7], sufficient_context=True), 200, 20),
    )
    body = client.post("/v1/query", json={"query": "Villa Aurora cancellation?"}, headers=HEADERS).json()
    assert body["answer_type"] == "insufficient_context"


def test_model_self_reported_insufficiency_is_respected(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda query, chunks: (GroundedAnswer(answer="Not sure.", citations=[1], sufficient_context=False), 200, 20),
    )
    body = client.post("/v1/query", json={"query": "Villa Aurora cancellation?"}, headers=HEADERS).json()
    assert body["answer_type"] == "insufficient_context"


def test_repeat_question_is_served_from_cache_for_free(client):
    payload = {"query": "Villa Aurora cancellation?"}
    client.post("/v1/query", json=payload, headers=HEADERS)
    second = client.post("/v1/query", json=payload, headers=HEADERS).json()

    assert second["usage"]["cached"] is True
    assert second["usage"]["estimated_cost_usd"] == 0.0


def test_missing_or_wrong_key_is_rejected(client):
    assert client.post("/v1/query", json={"query": "hello there"}).status_code == 401
    body = client.post("/v1/query", json={"query": "hello there"}, headers={"X-API-Key": "nope"}).json()
    assert body["error"]["code"] == "unauthorized"


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(main.settings, "rate_limit_per_minute", 2)
    for _ in range(2):
        client.get("/v1/documents", headers=HEADERS)

    response = client.get("/v1/documents", headers=HEADERS)
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert response.json()["error"]["code"] == "rate_limited"


def test_validation_errors_use_the_same_envelope(client):
    body = client.post("/v1/query", json={"query": "hi"}, headers=HEADERS).json()
    assert body["error"]["code"] == "invalid_request"
    assert "query" in body["error"]["message"]


def test_health_needs_no_key(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks_indexed"] == 1


def _boom(*args, **kwargs):
    raise AssertionError("the answer model must not be called when nothing is relevant")
