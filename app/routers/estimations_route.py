from datetime import datetime

from fastapi import APIRouter

from app.config import settings
from app.services.llm_service import generate_estimation
from app.schemas.request_io import EstimationRequest, EstimationResponse



router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    """Generating a software project estimation based on the provided meeting transcription."""
    llm_estimation = generate_estimation(request.transcription)

    return EstimationResponse(
        estimation=llm_estimation.estimation,
        model=llm_estimation.provider_info.model,
        provider=llm_estimation.provider_info.provider,
        timestamp=datetime.utcnow(),
    )
