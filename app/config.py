"""Settings and price table. Everything tunable lives here."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- credentials -------------------------------------------------------
    gemini_api_key: str = ""
    # Comma-separated list of keys this API accepts from callers.
    api_keys: str = "demo-key"

    # --- models ------------------------------------------------------------
    embed_model: str = "gemini-embedding-001"
    chat_model: str = "gemini-3.5-flash"
    # gemini-embedding-001 defaults to 3072 dims; 768 keeps the index 4x smaller
    # at close to the same retrieval quality (see README > Tradeoffs).
    embed_dim: int = 768

    # --- retrieval ---------------------------------------------------------
    dense_k: int = 20          # candidates from vector search
    bm25_k: int = 20           # candidates from lexical search
    rerank_k: int = 15         # fused candidates sent to the reranker
    top_k: int = 5             # chunks handed to the answer model
    rrf_k: int = 60            # Reciprocal Rank Fusion constant
    min_rerank_score: float = 0.35   # below this, nothing is "relevant enough"

    # --- serving -----------------------------------------------------------
    cache_ttl_seconds: int = 900
    # Free-tier keys are rate limited hard; back off and retry on 429.
    retry_base_seconds: float = 8.0
    rate_limit_per_minute: int = 20

    index_dir: str = "index"

    # --- pricing (USD per 1M tokens) ---------------------------------------
    # List prices for the models above. Update these whenever chat_model changes,
    # or the cost reported on every response silently drifts from reality.
    price_embed_input: float = 0.15
    price_chat_input: float = 0.30
    price_chat_output: float = 2.50

    @property
    def allowed_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
