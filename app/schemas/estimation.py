"""Request and response models for the estimation endpoint.

Session 4 contract: typed form-style request maps to a typed, validated
``EstimationResult`` (structured output via Instructor + Pydantic). Two model
validators enforce business rules that the LLM cannot break:

1. The cost of all phases must sum to ``total_cost_eur``.
2. Low-confidence answers (< 30%) must declare it explicitly by starting the
   summary with ``"Out of scope:"``.

When the LLM violates a validator, Instructor re-prompts the model with the
``ValueError`` message until it agrees (up to ``max_retries`` attempts).
"""

from enum import Enum

from app.schemas.llm_io import TokenUsage
from pydantic import BaseModel, Field, model_validator


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"





# --- Structured response ----------------------------------------------------


OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """One phase in the breakdown of an estimation."""

    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    """Structured estimation. The two validators below are the business rules
    that the LLM cannot break — Instructor will re-prompt the model when one
    of them raises.

    Field order is deliberate: ``phases`` comes BEFORE the totals so the LLM
    commits to the per-phase numbers first (autoregressive generation) and
    then only needs to sum them when filling the totals. Putting totals first
    leads the model to pick a round number and then back-fit phases to it,
    which it does very badly arithmetically — particularly with smaller
    models like ``gpt-4o-mini``.
    """

    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="after")
    def phases_sum_matches_total(self) -> "EstimationResult":
        phase_sum = sum(p.cost_eur for p in self.phases)
        if phase_sum != self.total_cost_eur:
            raise ValueError(
                f"phases sum ({phase_sum} EUR) does not match total_cost_eur "
                f"({self.total_cost_eur} EUR); adjust either the phases or the total"
            )
        return self

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self



class StructureCheck(BaseModel):
    """Level-1 structural evaluation of the generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]

# ESTIMATION REQUEST / RESPONSE MODELS --------------------------------------------------------------------

class EstimationRequest(BaseModel):
    """Typed payload sent by the business backend or Streamlit form."""

    description: str = Field(
        min_length=20,
        max_length=80000,
        description="Free-text description or transcription of the project to estimate.",
    )
    project_type: ProjectType = Field(description="Coarse-grained project category.")
    detail_level: DetailLevel = Field(description="How deep the estimation should go.")
    output_format: OutputFormat = Field(description="Shape of the rendered estimation.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional conversational session identifier (UUID v4 issued by "
            "POST /sessions). When provided, the server resumes the matching "
            "Session so history and project_metadata persist across turns."
        ),
    )


class AttachmentExtraction(BaseModel):
    """Per-file extraction trace returned to the caller for inspection.

    The text here is exactly what was concatenated into the description
    before the prompt was rendered — so the UI can show "what the LLM
    actually saw" for every uploaded attachment.
    """

    filename: str
    bytes: int = Field(ge=0, description="Original file size in bytes.")
    chars: int = Field(ge=0, description="Length of the extracted text in characters.")
    text: str = Field(default="", description="Full extracted text, identical to the chunk appended to the prompt.")


class EstimationResponse(BaseModel):
    result: EstimationResult
    prompt_version: str
    cached: bool = False
    # -- Extras about the model resolution and estimation metadata ---
    model: str = Field(default="", description="Name of the LLM model used for estimation")
    provider: str = Field(default="", description="Provider of the estimation LLM service")
    usage: TokenUsage = Field(default=None, description="Token usage information")
    finish_reason: str = Field(default="", description="Stop reason reported by the provider")
    latency_ms: int = Field(default=0, description="Server-side total latency in milliseconds")
    cost_usd: float = Field(default=0.0, description="Estimated USD cost based on token usage")
    attachments: list[AttachmentExtraction] = Field(
        default_factory=list,
        description=(
            "Per-file extraction trace, populated only for calls that "
            "processed attachments. Empty list on plain-transcript "
            "requests. Lets clients see exactly which text was injected "
            "into the prompt for each uploaded document."
        ),
    )