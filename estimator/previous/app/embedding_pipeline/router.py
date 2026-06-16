"""FastAPI router for the embedding ingest endpoint.

Orchestrates chunking + embedding for a batch of historical budgets and
returns the embedded chunks together with aggregate stats. Validation
errors are surfaced by FastAPI as 422; embedding-API failures are caught
here and turned into a 500 with a generic client message, while the
exception detail goes to structlog.
"""

import structlog
from fastapi import APIRouter, HTTPException

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import (
    IngestRequest,
    IngestResponse,
    IngestStats,
)


log = structlog.get_logger()

router = APIRouter(tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Chunk and embed a batch of historical budgets."""

    chunker = JSONStructuralChunker()
    chunks = chunker.chunk(request.budgets)

    log.info(
        "embedding_ingest.chunked",
        total_budgets=len(request.budgets),
        total_chunks=len(chunks),
    )

    try:
        embedder = OpenAIEmbedder()
        embedded = embedder.embed_many(chunks)
    except Exception as exc:
        log.error(
            "embedding_ingest.embedding_failed",
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Embedding service failed; check server logs for details.",
        ) from exc

    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=embedder.last_total_tokens,
        estimated_cost_usd=embedder.last_estimated_cost_usd,
    )

    log.info(
        "embedding_ingest.completed",
        total_budgets=stats.total_budgets,
        total_chunks=stats.total_chunks,
        total_tokens=stats.total_tokens,
        estimated_cost_usd=stats.estimated_cost_usd,
    )

    return IngestResponse(chunks=embedded, stats=stats)
