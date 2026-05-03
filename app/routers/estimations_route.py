from datetime import datetime
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.services.llm_service import generate_estimation, generate_estimation_stream
from app.schemas.request_io import EstimationRequest, EstimationResponse



router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    """Generating a software project estimation based on the provided meeting transcription."""

#log the incoming request for debugging
    print(f"Received estimation request at {datetime.utcnow()}: {request}")

    llm_estimation = generate_estimation(request.transcription)

    return EstimationResponse(
        estimation=llm_estimation.estimation,
        model=llm_estimation.provider_info.model,
        provider=llm_estimation.provider_info.provider,
        timestamp=datetime.utcnow(),
    )


@router.post("/estimate/stream")
async def estimate_stream(request: EstimationRequest) -> StreamingResponse:
    """Stream model output as it is generated for OpenAI estimations."""
    if settings.LLM_PROVIDER.lower() != "openai":
        raise HTTPException(
            status_code=400,
            detail="Streaming solo está disponible para el proveedor OpenAI."
        )

    def event_generator():
        for event in generate_estimation_stream(request.transcription):
            payload = json.dumps(event)
            yield f"event: {event.get('type', 'message')}\n"
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
