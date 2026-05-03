from datetime import datetime
from pydantic import BaseModel, Field


class EstimationRequest(BaseModel):
    transcription: str = Field(..., min_length=1, description="Texto de la transcripción de la reunión")


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    timestamp: datetime
