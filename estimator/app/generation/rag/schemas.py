"""Pydantic models for the embedding pipeline.

Input side mirrors the normalized historical-budget JSON (a budget with a list
of components). Output side carries chunks ready to embed and, once embedded,
the vectors plus aggregate stats.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Closed universe of client sectors present in the sample dataset. Kept as a
# Literal so a typo or an unexpected sector fails validation loudly instead of
# silently leaking into the metadata.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    """Who the budget belongs to. Travels as filterable context, not embedded."""

    name: str = Field(description="Client company name.")
    sector: Sector = Field(description="Client business sector.")
    country: str = Field(description="ISO-ish country code, e.g. 'ES'.")


class BudgetComponent(BaseModel):
    """A single line item of a historical budget."""

    component_id: str = Field(description="Stable id within the budget, e.g. 'AUTH-001'.")
    name: str = Field(description="Short human-readable component name.")
    description: str = Field(description="Detailed description of the work.")
    tech_stack: list[str] = Field(
        default_factory=list, description="Technologies involved in this component."
    )
    estimated_hours: int = Field(ge=0, description="Hours estimated for this component.")
    complexity: Complexity = Field(description="Coarse complexity bucket.")
    dependencies: list[str] = Field(
        default_factory=list, description="component_ids this one depends on."
    )


class Budget(BaseModel):
    """A complete historical budget with its components."""

    budget_id: str = Field(description="Stable budget id, e.g. 'BUD-2024-014'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(description="One-line summary of the project.")
    main_technology: str = Field(description="Primary technology / stack of the project.")
    year: int = Field(ge=2000, le=2100, description="Year the budget was produced.")
    total_estimated_hours: int = Field(ge=0, description="Sum of component hours, as recorded.")
    components: list[BudgetComponent] = Field(min_length=1, description="Budget line items.")


class Chunk(BaseModel):
    """A fragment ready to be embedded.

    ``text`` is what gets sent to the embeddings API; ``metadata`` carries
    filterable fields that travel alongside the chunk but are NOT embedded.
    """

    chunk_id: str = Field(description="Traceable id, format '{budget_id}::{component_id}'.")
    text: str = Field(description="Embeddable text: parent context + component detail.")
    metadata: dict = Field(default_factory=dict, description="Filterable, non-embedded fields.")
    token_count: int = Field(ge=0, description="Token count of ``text`` (tiktoken).")


class EmbeddedChunk(Chunk):
    """A :class:`Chunk` with its embedding vector attached."""

    embedding: list[float] = Field(
        description="Dense embedding vector (1536 dims for text-embedding-3-small)."
    )


class IngestRequest(BaseModel):
    """Payload for ``POST /embeddings/ingest`` (Session 8: persisting contract).

    One request = one document. ``content`` is the full budget JSON, validated
    against :class:`Budget` so a malformed corpus fails with a 422 before
    touching the database or the embeddings API.
    """

    source_path: str = Field(
        min_length=1, description="Provenance of the document, unique per ingest."
    )
    document_type: str = Field(
        min_length=1, max_length=50, description="Document family, e.g. 'historical_budget'."
    )
    content: Budget = Field(description="Full budget JSON, as produced upstream.")


class IngestResponse(BaseModel):
    """Response for ``POST /embeddings/ingest``: identifiers + ingest metrics.

    Vectors no longer travel over HTTP — they are persisted in pgvector.
    """

    document_id: int = Field(description="Primary key of the persisted document.")
    chunks_created: int = Field(ge=0, description="Chunks persisted for this document.")
    embedding_dimension: int = Field(description="Dimensionality of the stored vectors.")
    ingestion_time_ms: int = Field(ge=0, description="Wall-clock ingest time.")


class SearchRequest(BaseModel):
    """Payload for ``POST /search``."""

    query: str = Field(min_length=1, description="Free-text semantic query.")
    k: int = Field(default=5, ge=1, le=50, description="Number of nearest chunks to return.")


class SearchHit(BaseModel):
    """One ranked chunk. ``chunk_id`` is the DB primary key; the traceable
    corpus id ('BUD-X::COMP-Y' parts) travels inside ``metadata``."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float = Field(description="Cosine distance (lower = more similar).")
    metadata: dict


class SearchResponse(BaseModel):
    """Response for ``POST /search``."""

    query: str
    k: int
    search_time_ms: int = Field(ge=0)
    results: list[SearchHit]
