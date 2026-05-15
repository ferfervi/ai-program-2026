from __future__ import annotations

import time
import structlog
from typing import Iterator

from app.config import get_settings
from app.services.open_ai_service import  OpenAIEstimator
from app.services.anthropic_service import AnthropicEstimator


import time
import structlog

from app.schemas.llm_io import LLMInputModel




log = structlog.get_logger()

DEFAULT_MAX_TOKENS = 4000
EXTRACTION_MAX_TOKENS = 1500

from app.schemas.llm_io import LLMEstimation, LLMProviderInfo, LLMServiceError, TokenUsage


class LLMServiceError(Exception):
    """Raised when the LLM provider call fails."""


# ---------------------------------------------------------------------------
# Prompt building blocks
#
# The two ACTIVE_OUTPUT_PROMPT variants live side by side so the instructor
# can switch between them in the live session (Block 3.4) by editing the
# ACTIVE_OUTPUT_PROMPT assignment below. Uvicorn `--reload` picks up the
# change automatically.
# ---------------------------------------------------------------------------

PROMPT_OUTPUT_BASIC = "Generate an estimation for the project described above."

PROMPT_OUTPUT_STRUCTURED = """\
Generate the estimation with this exact structure:

## Project summary
[2-3 sentences describing the project scope and goals]

## Task breakdown
| Task | Hours | Cost (EUR) |
[one row per task; cost = hours * 62.50 EUR for developer tasks]

## Totals
- Total hours: [number]
- Total cost: [number] EUR
- Recommended team: [composition]
- Estimated duration: [weeks]

## Risks and assumptions
- [3-5 bullet points covering technical risks, scope assumptions, and external dependencies]
"""

# >>> Block 3.4 live switch: change the right-hand side to PROMPT_OUTPUT_STRUCTURED
ACTIVE_OUTPUT_PROMPT = PROMPT_OUTPUT_BASIC


INLINE_CLEANING_BLOCK = """\
The transcription you receive is from a real meeting and may contain:
- Informal small talk you must ignore
- Implicit requirements you must surface explicitly
- Contradictions where you must trust the most recent statement
- Non-technical jargon you must interpret

Extract ONLY the functional and technical requirements relevant to the estimation."""


EXTRACTION_SYSTEM_PROMPT = (
    "You are an analyst. Read the meeting transcription and produce a clean, "
    "deduplicated bullet list of functional requirements, non-functional "
    "requirements, integrations, constraints and explicit deadlines. Ignore "
    "fillers, divagations and off-topic remarks. Output Markdown only."
)


def generate_estimation(llm_input: LLMInputModel) -> LLMEstimation:
    """Genera una estimación a partir de la transcripción usando el proveedor configurado."""

    t0 = time.perf_counter()
    system_prompt = llm_input.system
    user_prompt = llm_input.user

    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()

    log.info("LLMService -generate estimation request", provider=provider, model=settings.LLM_MODEL)

    match provider:
        case "openai":
            openai_estimation = OpenAIEstimator().estimate(system_prompt, user_prompt)

            return LLMEstimation(
                estimation=openai_estimation.estimation,
                provider_info=openai_estimation.provider_info,
                token_usage=openai_estimation.token_usage,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                finish_reason=openai_estimation.finish_reason
            )
        
        case "anthropic":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            anthropic_estimation = AnthropicEstimator().estimate(messages)
            return LLMEstimation(
                estimation=anthropic_estimation.estimation,
                provider_info=anthropic_estimation.provider_info,
                token_usage=anthropic_estimation.token_usage,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                finish_reason="unknown"  # Placeholder for finish reason
            )
        
        
        case _:
            raise ValueError(
                f"Proveedor de LLM desconocido: {settings.LLM_PROVIDER}. Use 'openai' o 'anthropic'."
            )


def generate_estimation_stream(llm_input: LLMInputModel) -> Iterator[dict]:
    """Stream OpenAI estimation events, including partial text and final token usage."""
    system_prompt = llm_input.system
    user_prompt = llm_input.user

    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()
    log.info("LLMService stream - generate estimation stream request", provider=provider, model=settings.LLM_MODEL)

    match provider:
        case "openai":
            yield from OpenAIEstimator().estimate_stream(system_prompt, user_prompt)

        case "anthropic":
            raise ValueError("Streaming sólo está disponible para el proveedor OpenAI & lite_llm.")
                

        case _:
            raise ValueError(
                f"Proveedor de LLM desconocido: {settings.LLM_PROVIDER}. Use 'openai' o 'anthropic'."
            )
