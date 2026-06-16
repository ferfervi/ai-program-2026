"""Pydantic v2 models for the embedding pipeline.

Mirrors the shape of `data/budgets_sample.json`: a list of
historical budgets, each one with structured client metadata and a list
of typed components. Downstream we chunk each component into an
embeddable text plus a small filterable metadata dict, embed it, and
return the embedded chunks together with aggregate stats.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Sector = Literal["fintech", "e-commerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    """Client-side metadata attached to a budget."""

    name: str = Field(min_length=1, max_length=128)
    sector: Sector = Field(description="Closed list of business sectors we ingest.")
    country: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code (e.g. 'ES', 'DE').",
    )

    @model_validator(mode="after")
    def country_is_uppercase(self) -> "ClientMetadata":
        if self.country != self.country.upper():
            raise ValueError(
                f"country must be uppercase ISO 3166-1 alpha-2; got {self.country!r}"
            )
        return self


class BudgetComponent(BaseModel):
    """One line-item component inside a budget."""

    component_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    tech_stack: list[str] = Field(
        min_length=1,
        description="Technologies involved in this component (free-form slugs).",
    )
    estimated_hours: int = Field(ge=1, le=100_000)
    complexity: Complexity
    dependencies: list[str] = Field(
        default_factory=list,
        description="component_id values this component depends on.",
    )


class Budget(BaseModel):
    """A full historical budget, parent of N components."""

    budget_id: str = Field(min_length=1, max_length=64)
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1, max_length=2000)
    main_technology: str = Field(min_length=1, max_length=64)
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: int = Field(ge=1, le=1_000_000)
    components: list[BudgetComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def components_hours_match_total(self) -> "Budget":
        components_sum = sum(c.estimated_hours for c in self.components)
        if components_sum != self.total_estimated_hours:
            raise ValueError(
                f"components estimated_hours sum ({components_sum}) does not match "
                f"total_estimated_hours ({self.total_estimated_hours}) for budget "
                f"{self.budget_id!r}"
            )
        return self


class Chunk(BaseModel):
    """A text fragment ready to be embedded."""

    chunk_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    metadata: dict = Field(
        default_factory=dict,
        description="Filterable metadata (e.g. sector, year, complexity, budget_id).",
    )
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    """A chunk plus the embedding vector returned by the embedding model."""

    embedding: list[float] = Field(min_length=1)


class IngestStats(BaseModel):
    """Aggregate stats for one ingest run."""

    total_budgets: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)


class IngestRequest(BaseModel):
    """Payload accepted by the ingest endpoint."""

    budgets: list[Budget] = Field(min_length=1)


class IngestResponse(BaseModel):
    """Result returned by the ingest endpoint."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats
