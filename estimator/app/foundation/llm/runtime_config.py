"""Runtime-mutable model configuration, backed by Redis.

``Settings`` (.env) stays the immutable layer of *defaults*; this store holds
runtime *overrides* as deltas in a Redis hash, so the instructor can switch
models from the Settings UI mid-session without editing .env or recreating
containers. Overrides survive ``--reload`` restarts and are shared across
workers (Redis is the source of truth, read on every call).

Failure semantics:
- Reads degrade gracefully: if Redis is down, ``effective()`` falls back to
  the settings default (the service behaves exactly as configured by .env).
- Writes re-raise (`RuntimeConfigUnavailable`): a failed override must be
  visible to the caller — the API maps it to a 503.
"""

from __future__ import annotations

import redis
import structlog

from app.config import Settings

log = structlog.get_logger()

# The LLM model knobs that can be overridden at runtime. Each name is also a
# ``Settings`` field (the default). EMBEDDING_MODEL is deliberately absent:
# changing it would invalidate every stored vector, so it stays .env-only.
MODEL_KEYS: tuple[str, ...] = (
    "PRIMARY_MODEL",
    "FALLBACK_MODEL",
    "CRITIC_MODEL",
    "METADATA_EXTRACTOR_MODEL",
    "COMPRESSION_MODEL",
    "PROPOSITIONAL_CHUNKER_MODEL",
    "CONTEXTUAL_CHUNKER_MODEL",
)

HASH_KEY = "estimator:runtime_config"


class RuntimeConfigUnavailable(Exception):
    """Raised when an override WRITE fails (Redis unreachable)."""


class RuntimeModelConfig:
    """Redis-hash-backed override store for the LLM model knobs."""

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeModelConfig":
        return cls(redis.from_url(url, decode_responses=True), settings)

    def get(self, key: str) -> str | None:
        """Return the override for ``key``, or ``None`` (unset / Redis down)."""
        self._validate_key(key)
        try:
            return self._redis.hget(HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key=key, error=str(exc)[:200])
            return None

    def set(self, key: str, value: str | None) -> None:
        """Set the override for ``key``; ``None`` clears it (back to default)."""
        self._validate_key(key)
        try:
            if value is None:
                self._redis.hdel(HASH_KEY, key)
            else:
                self._redis.hset(HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective(self, key: str) -> str:
        """The value the pipeline should use: override if set, else the default."""
        return self.get(key) or self.default(key)

    def default(self, key: str) -> str:
        self._validate_key(key)
        return getattr(self._settings, key)

    def is_overridden(self, key: str) -> bool:
        return self.get(key) is not None

    def snapshot(self) -> dict[str, dict[str, str | bool]]:
        """One entry per model key: ``{effective, default, overridden}``."""
        try:
            overrides = self._redis.hgetall(HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            key: {
                "effective": overrides.get(key) or self.default(key),
                "default": self.default(key),
                "overridden": key in overrides,
            }
            for key in MODEL_KEYS
        }

    def reset_all(self) -> None:
        """Drop every override (back to .env defaults)."""
        try:
            self._redis.delete(HASH_KEY)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    @staticmethod
    def _validate_key(key: str) -> None:
        if key not in MODEL_KEYS:
            raise ValueError(f"Unknown model key: {key}")


# --- Session 10: runtime toggles for the retrieval pipeline ------------------
# Same Redis-hash override pattern as the model knobs, but for the two retrieval
# switches the live session flips mid-demo (search mode and reranking). Kept in a
# SEPARATE hash so resetting the model config does not clear them and vice versa.
# The settings fields are the immutable defaults: RETRIEVAL_SEARCH_MODE ("vector"
# | "hybrid") and RERANKER_ENABLED (bool).
RETRIEVAL_HASH_KEY = "estimator:runtime_retrieval"

SEARCH_MODE_KEY = "RETRIEVAL_SEARCH_MODE"
RERANK_KEY = "RERANKER_ENABLED"
RETRIEVAL_KEYS: tuple[str, ...] = (SEARCH_MODE_KEY, RERANK_KEY)

_VALID_SEARCH_MODES = ("vector", "hybrid")


class RuntimeRetrievalConfig:
    """Redis-hash-backed override store for the Session 10 retrieval toggles.

    Reads degrade to the settings default if Redis is down (the pipeline keeps
    serving exactly as .env configured it); writes re-raise so a failed override
    is visible to the API (mapped to 503), matching :class:`RuntimeModelConfig`.
    """

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> "RuntimeRetrievalConfig":
        return cls(redis.from_url(url, decode_responses=True), settings)

    def _get_raw(self, key: str) -> str | None:
        try:
            return self._redis.hget(RETRIEVAL_HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key=key, error=str(exc)[:200])
            return None

    def _set_raw(self, key: str, value: str | None) -> None:
        try:
            if value is None:
                self._redis.hdel(RETRIEVAL_HASH_KEY, key)
            else:
                self._redis.hset(RETRIEVAL_HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective_search_mode(self) -> str:
        override = self._get_raw(SEARCH_MODE_KEY)
        if override in _VALID_SEARCH_MODES:
            return override
        return self._settings.RETRIEVAL_SEARCH_MODE

    def effective_rerank(self) -> bool:
        override = self._get_raw(RERANK_KEY)
        if override is None:
            return self._settings.RERANKER_ENABLED
        return override.lower() == "true"

    def set_search_mode(self, value: str | None) -> None:
        if value is not None and value not in _VALID_SEARCH_MODES:
            raise ValueError(f"Invalid search mode: {value}")
        self._set_raw(SEARCH_MODE_KEY, value)

    def set_rerank(self, value: bool | None) -> None:
        self._set_raw(RERANK_KEY, None if value is None else str(value).lower())

    def snapshot(self) -> dict[str, dict[str, object]]:
        """``{effective, default, overridden}`` per retrieval toggle (for the UI)."""
        try:
            overrides = self._redis.hgetall(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            SEARCH_MODE_KEY: {
                "effective": self.effective_search_mode(),
                "default": self._settings.RETRIEVAL_SEARCH_MODE,
                "overridden": SEARCH_MODE_KEY in overrides,
            },
            RERANK_KEY: {
                "effective": self.effective_rerank(),
                "default": self._settings.RERANKER_ENABLED,
                "overridden": RERANK_KEY in overrides,
            },
        }
