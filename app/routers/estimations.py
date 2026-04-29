from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm_service import generate_estimation


class EstimationRequest(BaseModel):
    transcription: str = Field(..., min_length=1, description="Texto de la transcripción de la reunión")


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    timestamp: datetime


router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
def estimate(request: EstimationRequest) -> EstimationResponse:
    estimation_text = generate_estimation(request.transcription)
    provider = settings.LLM_PROVIDER.lower()
    model = settings.LLM_MODEL
    return EstimationResponse(
        estimation=estimation_text,
        model=model,
        provider=provider,
        timestamp=datetime.utcnow(),
    )
