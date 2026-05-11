from datetime import datetime
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.services.llm_service import generate_estimation, generate_estimation_stream
from app.schemas.request_io import EstimationRequest, EstimationResponse
from app.schemas.llm_io import LLMInputModel
from app.prompts.loader import render_estimation_prompt


import structlog
log = structlog.get_logger()

router = APIRouter(tags=["estimations"])


def _log_stream_event(event: dict) -> None:
    event_type = event.get("type", "unknown")
    if event_type == "done":
        usage = event.get("token_usage", {})
        log.info(
            "stream_response_completed",
            model=event.get("model"),
            provider=event.get("provider"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cost_usd=usage.get("cost_usd"),
            latency_ms=event.get("latency_ms"),
            cache_hit=event.get("cache_hit"),
        )
    elif event_type == "delta":
        log.debug("stream_delta_received", text_length=len(event.get("text", "")))
    else:
        log.warning("stream_unknown_event_type", event_type=event_type)


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    """Generating a software project estimation based on the provided meeting transcription."""

    log.info("Received estimation request", request=request.model_dump())
    system, user = render_estimation_prompt(request)
    llm_input_model = LLMInputModel(system=system, user=user)
    
    llm_estimation = generate_estimation(llm_input=llm_input_model)

    return EstimationResponse(
        text=llm_estimation.estimation,
        model=llm_estimation.provider_info.model,
        provider=llm_estimation.provider_info.provider,
        usage=llm_estimation.token_usage,
        finish_reason=llm_estimation.finish_reason,
        latency_ms=llm_estimation.latency_ms, 
        cost_usd=llm_estimation.token_usage.cost_usd,
        cache_hit=llm_estimation.cache_hit
    )


@router.post("/estimate/stream")
async def estimate_stream(request: EstimationRequest) -> StreamingResponse:
    """Stream model output as it is generated for OpenAI estimations."""

    log.info("Router [/estimate/stream] request", request=request.model_dump())
    system, user = render_estimation_prompt(request)
    llm_input_model = LLMInputModel(system=system, user=user)

    settings = get_settings()
    if settings.LLM_PROVIDER.lower() != "openai" and settings.LLM_PROVIDER.lower() != "lite_llm":
        raise HTTPException(
            status_code=400,
            detail="Streaming solo está disponible para el proveedor OpenAI."
        )

    def event_generator():
        for event in generate_estimation_stream(llm_input=llm_input_model):
            _log_stream_event(event)
            payload = json.dumps(event)
            yield f"event: {event.get('type', 'message')}\n"
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
