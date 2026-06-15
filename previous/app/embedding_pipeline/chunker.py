"""Structural chunker for historical budgets.

One component = one chunk. We deliberately prepend the parent budget's
context (project summary, sector, year, main technology) into each
chunk's text so that a component like "Authentication backend" carries
the client/sector/tech context with it at retrieval time. This is the
"contextual chunk header" idea: keep the embedded text self-contained
so the vector lives near other components of similar scope, not just
similar wording.

No overlap, no fixed-size splitting. The JSON structure is the unit of
meaning — if a single component's description is exceptionally long,
we'd rather see it explicitly than silently fragment it.
"""

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk


EMBEDDING_MODEL = "text-embedding-3-small"


class JSONStructuralChunker:
    """Turn a list of Budgets into one Chunk per BudgetComponent."""

    def __init__(self) -> None:
        self._encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                chunks.append(self._chunk_component(budget, component))
        return chunks

    def _chunk_component(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = self._render_text(budget, component)
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata={
                "budget_id": budget.budget_id,
                "component_id": component.component_id,
                "client_sector": budget.client_metadata.sector,
                "main_technology": budget.main_technology,
                "year": budget.year,
                "complexity": component.complexity,
                "estimated_hours": component.estimated_hours,
            },
            token_count=len(self._encoding.encode(text)),
        )

    @staticmethod
    def _render_text(budget: Budget, component: BudgetComponent) -> str:
        return (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {budget.client_metadata.sector} | "
            f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
            f"\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
