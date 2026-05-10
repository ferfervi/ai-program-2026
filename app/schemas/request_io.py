from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal

from pydantic import BaseModel, Field
from app.schemas.llm_io import TokenUsage
from app.schemas.estimation_io import StructureCheck

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]



class EstimationRequest(BaseModel):
    transcription: str = Field(..., min_length=1, description="Texto de la transcripción de la reunión")


class EstimationResponse(BaseModel):
    estimation: str = Field(..., description="Generated software estimation in markdown")
    model: str = Field(..., description="Name of the LLM model used for estimation")
    provider: str = Field(..., description="Provider of the estimation LLM service")
    usage: TokenUsage = Field(..., description="Token usage information")
    finish_reason: str = Field(..., description="Stop reason reported by the provider")
    preprocessing: PreprocessingMode = "none"
    extracted_requirements: str | None = Field(
        default=None,
        description="Phase-1 output when preprocessing='two_phase'; null otherwise",
    )
    latency_ms: int = Field(..., description="Server-side total latency in milliseconds")
    validation: StructureCheck | None = None

    # --- Session 3 — wrapper metadata (additive, defaults preserve Session 2 tests) ---
    cache_hit: bool = Field(default=False, description="True when the response came from Redis")
    cost_usd: float = Field(default=0.0, description="Estimated USD cost based on token usage")