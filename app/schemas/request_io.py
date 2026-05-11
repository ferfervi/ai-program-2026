from datetime import datetime
from pydantic import BaseModel, Field

from pydantic import BaseModel, Field
from app.schemas.llm_io import TokenUsage
from app.schemas.estimation_io import StructureCheck, PreprocessingMode, ProjectType, DetailLevel, OutputFormat




class EstimationRequest(BaseModel):
    description: str = Field(..., min_length=20, max_length=2000, description="Transcription of the project requirements meeting")
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat


class EstimationResponse(BaseModel):
    text: str = Field(..., description="Generated software estimation in markdown")
    # -- Extras about the model resolution and estimation metadata ---
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