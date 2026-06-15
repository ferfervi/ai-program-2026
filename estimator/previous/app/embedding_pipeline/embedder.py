"""OpenAI embedder for the embedding pipeline.

Wraps `text-embedding-3-small` with batched calls, simple exponential
backoff on rate limiting, structlog instrumentation per batch, and
input-token cost tracking. The pricing constant is module-level and
labelled so it is obvious where to update it when OpenAI changes prices.
"""

import time

import structlog
from openai import OpenAI, RateLimitError

from app.config import get_settings
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk


# As of 2025-Q4, OpenAI lists text-embedding-3-small at $0.02 / 1M input
# tokens. This rate changes over time — update here when it does.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PRICE_USD_PER_MILLION_INPUT_TOKENS = 0.02

BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)

log = structlog.get_logger()


class OpenAIEmbedder:
    """Batched OpenAI embedder with retry-on-rate-limit and cost tracking.

    After `embed_many` runs, aggregate stats for the last call are
    exposed on the instance so the caller can build the response stats:
    `last_total_tokens` and `last_estimated_cost_usd`.
    """

    def __init__(self, client: OpenAI | None = None) -> None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self.client = client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.last_total_tokens: int = 0
        self.last_estimated_cost_usd: float = 0.0

    def embed_one(self, text: str) -> list[float]:
        response = self._create_with_retry([text])
        return response.data[0].embedding

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        total_tokens = 0

        for batch_index, start in enumerate(range(0, len(chunks), BATCH_SIZE)):
            batch = chunks[start : start + BATCH_SIZE]
            t0 = time.perf_counter()
            response = self._create_with_retry([c.text for c in batch])
            latency_ms = int((time.perf_counter() - t0) * 1000)

            batch_tokens = getattr(response.usage, "total_tokens", 0) or 0
            total_tokens += batch_tokens

            log.info(
                "openai_embedder.batch",
                batch_index=batch_index,
                batch_size=len(batch),
                total_tokens=batch_tokens,
                latency_ms=latency_ms,
            )

            for chunk, item in zip(batch, response.data):
                embedded.append(
                    EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        token_count=chunk.token_count,
                        embedding=item.embedding,
                    )
                )

        self.last_total_tokens = total_tokens
        self.last_estimated_cost_usd = self._estimate_cost(total_tokens)
        return embedded

    def _create_with_retry(self, inputs: list[str]):
        last_error: RateLimitError | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=inputs,
                )
            except RateLimitError as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                wait_s = RETRY_BACKOFF_SECONDS[attempt]
                log.warning(
                    "openai_embedder.rate_limited",
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    backoff_seconds=wait_s,
                )
                time.sleep(wait_s)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _estimate_cost(total_tokens: int) -> float:
        cost = (total_tokens / 1_000_000) * EMBEDDING_PRICE_USD_PER_MILLION_INPUT_TOKENS
        return round(cost, 8)
