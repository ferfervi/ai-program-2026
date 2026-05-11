"""FastAPI dependency factories for shared singletons (cache and LLM wrapper)."""

from __future__ import annotations

from functools import lru_cache

import structlog

from app.config import get_settings
from app.services.cache_service import EstimationCache
from app.services.litellm_wrapper_service import LiteLLMMWrapperService

log = structlog.get_logger()


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_litellm_wrapper() -> LiteLLMMWrapperService:
    settings = get_settings()

    openai_key = settings.OPENAI_API_KEY
    anthropic_key = settings.ANTHROPIC_API_KEY
    log.info(
        "litellm_wrapper_init",
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        openai_key_set=bool(openai_key),
        openai_key_prefix=openai_key[:8] if openai_key else None,
        anthropic_key_set=bool(anthropic_key),
        anthropic_key_prefix=anthropic_key[:8] if anthropic_key else None,
    )

    return LiteLLMMWrapperService(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
    )