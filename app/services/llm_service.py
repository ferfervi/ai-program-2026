from __future__ import annotations

import time
from datetime import datetime
import structlog
from typing import Iterator

from app.config import get_settings
from app.services.open_ai_service import  OpenAIEstimator
from app.services.anthropic_service import AnthropicEstimator
from app.context.examples import ESTIMATION_EXAMPLES


import time
from dataclasses import dataclass
from typing import Any

import structlog

from app.context.examples import format_examples_for_prompt, select_examples
from app.dependencies import get_litellm_wrapper
from app.schemas.estimation_io import ExampleFormat, PreprocessingMode




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


@dataclass
class GenerationOptions:
    """Per-request knobs that drive prompt construction and the LLM call."""

    preprocessing: PreprocessingMode = "none"
    example_format: ExampleFormat = "markdown"
    num_examples: int = 3
    use_examples: bool = True
    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget: int | None = None


# def build_system_prompt() -> str:
#     """Construye el prompt de sistema con el rol y los ejemplos de estimación."""
#     instructions = (
#         "Eres un estimador de software experto. Genera una estimación detallada y práctica "
#         "basada en ejemplos previos y en la transcripción de una nueva reunión. "
#         "Usa los ejemplos como referencia del formato y la profundidad esperada. "
#         "Tu respuesta debe incluir un desglose de tareas, un total estimado, equipo recomendado "
#         "y duración estimada."
#     )

#     examples = []
#     for index, example in enumerate(ESTIMATION_EXAMPLES, start=1):
#         summary = example.get("meeting_summary", "").strip()
#         estimation = example.get("estimation", "").strip()
#         examples.append(
#             f"Ejemplo {index}:\nResumen de la reunión: {summary}\nEstimación:\n{estimation}"
#         )

#     return f"{instructions}\n\n" + "\n\n".join(examples)

# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def build_system_prompt(
    example_format: ExampleFormat = "markdown",
    num_examples: int = 2,
    use_examples: bool = True,
    inline_cleaning: bool = False,
) -> str:
    """Assemble the system prompt with role, rates, output spec and (optionally) examples."""
    role = (
        "You are a senior software consultant with 15+ years of experience in project "
        "estimation. Your task is to produce a detailed software project estimation based "
        "on a meeting transcription provided by the user."
    )
    rates = (
        "Use a developer rate of approximately 62.50 EUR/hour (500 EUR/day) and a designer "
        "rate of approximately 50 EUR/hour (400 EUR/day). Provide realistic, well-justified "
        "numbers."
    )

    examples_block = ""
    if use_examples and num_examples > 0:
        rendered = format_examples_for_prompt(select_examples(num_examples), example_format)
        if rendered:
            examples_block = (
                "Below are reference estimations from previous projects. Use them as a guide "
                "for structure, level of detail, and realistic pricing. Adapt the content to "
                "match the specific project described in the transcription.\n\n"
                + rendered
            )

    cleaning_block = INLINE_CLEANING_BLOCK if inline_cleaning else ""

    sections = [role, cleaning_block, rates, ACTIVE_OUTPUT_PROMPT, examples_block]
    return "\n\n".join(s for s in sections if s)



# LLM Wrapper LITE LLM Service
def _invoke_lite_llm(
    *,
    system_prompt: str,
    user_message: str,
    model_override: str | None,
    max_tokens: int,
    thinking_budget: int | None,
) -> dict[str, Any]:
    """Single seam through which every LLM call passes. Tests monkeypatch this."""
    wrapper = get_litellm_wrapper()
    return wrapper.complete(
        system_prompt=system_prompt,
        user_message=user_message,
        model_override=model_override,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
    )


# LLM Wrapper LITE LLM Service
def _invoke_lite_llm_stream(
    *,
    system_prompt: str,
    user_message: str,
    model_override: str | None,
    max_tokens: int,
    thinking_budget: int | None,
) -> Iterator[str | dict]:
    """Single seam through which every streamed LLM call passes. Tests monkeypatch this."""
    wrapper = get_litellm_wrapper()
    return wrapper.complete_stream(
        system_prompt=system_prompt,
        user_message=user_message,
        model_override=model_override,
        max_tokens=max_tokens,
    )

# ---------------------------------------------------------------------------
# Two-phase preprocessing (phase 1: requirement extraction)
# ---------------------------------------------------------------------------


def extract_requirements(
    transcription: str,
    opts: GenerationOptions,
) -> tuple[str, dict, float]:
    """Run the cheap phase-1 LLM call that turns a raw transcription into clean requirements.

    Returns ``(requirements_text, usage_dict, cost_usd)``.
    """
    log.info("extracting_requirements", model_override=opts.model)

    result = _invoke_lite_llm(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_message=transcription,
        model_override=opts.model,
        max_tokens=EXTRACTION_MAX_TOKENS,
        thinking_budget=opts.thinking_budget,
    )

    return (
        result["estimation"],
        {
            "input": result["usage"]["input_tokens"],
            "output": result["usage"]["output_tokens"],
        },
        float(result.get("cost_usd", 0.0)),
    )



def generate_estimation(transcript: str) -> LLMEstimation:
    """Genera una estimación a partir de la transcripción usando el proveedor configurado."""

    t0 = time.perf_counter()
    system_prompt = build_system_prompt()
    user_prompt = transcript.strip()

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
        
        case "lite_llm":

            prep_usage = {"input": 0, "output": 0}
            opts =  GenerationOptions()

            t0 = time.perf_counter()

            prep_usage = {"input": 0, "output": 0}
            prep_cost = 0.0
            extracted_requirements: str | None = None
            user_input = transcript

            if opts.preprocessing == "two_phase":
                extracted_requirements, prep_usage, prep_cost = extract_requirements(transcript, opts)
                user_input = extracted_requirements

            system_prompt = build_system_prompt(
                example_format=opts.example_format,
                num_examples=opts.num_examples,
                use_examples=opts.use_examples,
                inline_cleaning=(opts.preprocessing == "inline_cleaning"),
            )

            log.info(
                "LLMService - LiteLLM invocaction",
                model_override=opts.model,
                preprocessing=opts.preprocessing,
                example_format=opts.example_format,
                num_examples=opts.num_examples,
                use_examples=opts.use_examples,
                max_tokens=opts.max_tokens,
                thinking_budget=opts.thinking_budget,
            )

            try:
                result = _invoke_lite_llm(
                    system_prompt=system_prompt,
                    user_message=user_input,
                    model_override=None,
                    max_tokens=opts.max_tokens,
                    thinking_budget=opts.thinking_budget,
                )   

                log.info("llm_response result", **result)

                result["usage"]["preprocessing_input_tokens"] = prep_usage["input"]
                result["usage"]["preprocessing_output_tokens"] = prep_usage["output"]
                result["preprocessing"] = opts.preprocessing
                result["extracted_requirements"] = extracted_requirements
                result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                result["cost_usd"] = round(float(result.get("cost_usd", 0.0)) + prep_cost, 6)
                # ``cache_hit`` is whatever the wrapper returned for the main estimation call.
                result.setdefault("cache_hit", False)


                return LLMEstimation(
                    estimation=result["estimation"],
                    provider_info=LLMProviderInfo(provider=result["provider"], model=result["model"] or opts.model or "default"),
                    token_usage=TokenUsage(
                        input_tokens=result["usage"]["input_tokens"],
                        output_tokens=result["usage"]["output_tokens"],
                        total_tokens=result["usage"]["input_tokens"] + result["usage"]["output_tokens"],
                        cost_usd=result["cost_usd"],
                        preprocessing_input_tokens=result["usage"].get("preprocessing_input_tokens", 0),
                        preprocessing_output_tokens=result["usage"].get("preprocessing_output_tokens", 0),
                    ),
                    latency_ms=result["latency_ms"],
                    finish_reason=result.get("finish_reason", "unknown"),
                    # Session 3 additions:
                    cache_hit=result.get("cache_hit", False)
                )


            except Exception as exc:
                log.error("llm_call_failed", error=str(exc), error_type=type(exc).__name__)
                raise LLMServiceError(f"LLM call failed: {exc}") from exc

        
        case _:
            raise ValueError(
                f"Proveedor de LLM desconocido: {settings.LLM_PROVIDER}. Use 'openai' o 'anthropic'."
            )


def generate_estimation_stream(transcript: str) -> Iterator[dict]:
    """Stream OpenAI estimation events, including partial text and final token usage."""
    system_prompt = build_system_prompt()
    user_prompt = transcript.strip()

    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()
    log.info("LLMService stream - generate estimation stream request", provider=provider, model=settings.LLM_MODEL)

    match provider:
        case "openai":
            yield from OpenAIEstimator().estimate_stream(system_prompt, user_prompt)

        case "anthropic":
            raise ValueError("Streaming sólo está disponible para el proveedor OpenAI & lite_llm.")
        
        case "lite_llm":
            
            prep_usage = {"input": 0, "output": 0}
            opts =  GenerationOptions()

            t0 = time.perf_counter()

            prep_usage = {"input": 0, "output": 0}
            prep_cost = 0.0
            extracted_requirements: str | None = None
            user_input = transcript

            if opts.preprocessing == "two_phase":
                extracted_requirements, prep_usage, prep_cost = extract_requirements(transcript, opts)
                user_input = extracted_requirements

            system_prompt = build_system_prompt(
                example_format=opts.example_format,
                num_examples=opts.num_examples,
                use_examples=opts.use_examples,
                inline_cleaning=(opts.preprocessing == "inline_cleaning"),
            )

            log.info(
                "generating_estimation",
                model_override=opts.model,
                preprocessing=opts.preprocessing,
                example_format=opts.example_format,
                num_examples=opts.num_examples,
                use_examples=opts.use_examples,
                max_tokens=opts.max_tokens,
                thinking_budget=opts.thinking_budget,
            )

            try:
                estimation_text = ""
                stream_usage: dict = {}
                for item in _invoke_lite_llm_stream(
                    system_prompt=system_prompt,
                    user_message=user_input,
                    model_override=None,
                    max_tokens=opts.max_tokens,
                    thinking_budget=opts.thinking_budget,
                ):
                    if isinstance(item, dict):
                        stream_usage = item
                    elif item:
                        estimation_text += item
                        yield {
                            "type": "delta",
                            "text": item,
                        }

                yield {
                    "type": "done",
                    "estimation": estimation_text,
                    "provider": stream_usage.get("provider", "litellm"),
                    "model": stream_usage.get("model", opts.model or "default"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "token_usage": {
                        "input_tokens": stream_usage.get("input_tokens", 0),
                        "output_tokens": stream_usage.get("output_tokens", 0),
                        "total_tokens": stream_usage.get("total_tokens", 0),
                        "cost_usd": stream_usage.get("cost_usd", 0.0),
                    },
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "finish_reason": "stop",
                    "cache_hit": False,
                }
            except Exception as exc:
                log.error("llm_call_failed", error=str(exc), error_type=type(exc).__name__)
                raise LLMServiceError(f"LLM call failed: {exc}") from exc
                

        case _:
            raise ValueError(
                f"Proveedor de LLM desconocido: {settings.LLM_PROVIDER}. Use 'openai' o 'anthropic'."
            )
