"""Unit tests for post-generation validation (Session 9)."""

from __future__ import annotations

from app.generation.rag.schemas import (
    Estimate,
    RetrievedChunk,
    SourceCitation,
    TaskItem,
    WorkModule,
)
from app.generation.rag.validation import check_coherence, validate_citations


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content="Component: Auth\nEstimated hours: 120",
        sector="finance",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.3,
    )


def _estimate(
    *, source_ids: list[int], component_sources: list[int], confidence="high"
) -> Estimate:
    return Estimate(
        total_engineer_days=20,
        duration_weeks=4,
        modules=[
            WorkModule(
                name="Authentication",
                tasks=[TaskItem(name="Auth", engineer_days=20, sources=component_sources)],
            )
        ],
        sources=[
            SourceCitation(source_id=sid, relevance="primary", used_for="auth")
            for sid in source_ids
        ],
        assumptions=[],
        confidence=confidence,
        reasoning="Derived from retrieved budgets.",
    )


def test_validate_citations_all_valid_returns_empty():
    chunks = [_chunk(1), _chunk(2)]
    estimate = _estimate(source_ids=[1, 2], component_sources=[1])
    assert validate_citations(estimate, chunks) == []


def test_validate_citations_flags_fabricated_ids():
    chunks = [_chunk(1), _chunk(2)]
    # cites 99 (top-level) and 42 (inside a component) — neither retrieved.
    estimate = _estimate(source_ids=[1, 99], component_sources=[42])
    assert validate_citations(estimate, chunks) == [42, 99]


def test_validate_citations_no_sources_is_valid():
    chunks = [_chunk(1)]
    estimate = _estimate(source_ids=[], component_sources=[])
    assert validate_citations(estimate, chunks) == []


def test_validate_citations_empty_retrieval_flags_every_cited_id():
    estimate = _estimate(source_ids=[1], component_sources=[2])
    assert validate_citations(estimate, []) == [1, 2]


def test_check_coherence_insufficient_with_nulls_is_coherent():
    estimate = Estimate(
        total_engineer_days=None,
        duration_weeks=None,
        confidence="insufficient",
        reasoning="no sources",
        insufficient_context_explanation="No relevant budgets retrieved.",
    )
    assert check_coherence(estimate) is True


def test_check_coherence_insufficient_with_numbers_is_incoherent():
    estimate = Estimate(
        total_engineer_days=10,
        duration_weeks=2,
        confidence="insufficient",
        reasoning="contradiction",
        insufficient_context_explanation="",
    )
    assert check_coherence(estimate) is False


def test_check_coherence_non_insufficient_always_true():
    estimate = _estimate(source_ids=[1], component_sources=[1], confidence="low")
    assert check_coherence(estimate) is True
