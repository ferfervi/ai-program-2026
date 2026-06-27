"""Post-generation checks for grounded estimates (Session 9).

Two independent guards run after the LLM returns an :class:`Estimate`:

* :func:`validate_citations` — every cited ``source_id`` must correspond to a
  chunk that was actually retrieved. Fabricated ids are the classic grounding
  failure and trigger one corrective retry in the orchestrator.
* :func:`check_coherence` — the ``insufficient`` confidence level has a strict
  shape (no numbers, an explanation present); a violation is a malformed
  response, not a valid estimate.
"""

from __future__ import annotations

from app.generation.rag.schemas import Estimate, RetrievedChunk


def validate_citations(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
) -> list[int]:
    """Return the cited source ids that were never retrieved (fabricated).

    Checks both the top-level ``sources`` citations and the per-task
    ``modules[].tasks[].sources``. An empty list means every citation is valid
    (including the edge case of an estimate that cites nothing at all).

    Parameters
    ----------
    estimate:
        The generated estimate to inspect.
    retrieved_chunks:
        The chunks the estimate was supposed to be grounded in.

    Returns
    -------
    list[int]
        Sorted, de-duplicated fabricated source ids (empty if all valid).
    """
    valid_ids = {chunk.id for chunk in retrieved_chunks}

    cited_ids: set[int] = {citation.source_id for citation in estimate.sources}
    for module in estimate.modules:
        for task in module.tasks:
            cited_ids.update(task.sources)

    return sorted(cited_ids - valid_ids)


def check_coherence(estimate: Estimate) -> bool:
    """Return whether the estimate's confidence level matches its content.

    When ``confidence == "insufficient"``: both numeric totals must be ``None``,
    ``modules`` must be empty, and ``insufficient_context_explanation`` must be
    non-empty. Any other confidence level is always considered coherent here
    (the numeric checks belong to the schema/business rules, not to this guard).
    """
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )
