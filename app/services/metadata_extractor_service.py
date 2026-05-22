"""LLM-backed extractor for ``ProjectMetadata`` (Step 4).

After every turn we hand the extractor the *current* known metadata plus
the latest exchange (user transcript + assistant summary) and ask the model
to return the **updated full** ``ProjectMetadata``. Instructor enforces the
shape, so the call cannot return malformed data.

Why an LLM extractor (not regex heuristics):
    - Transcripts arrive in Spanish and English and mix capitalisation,
      abbreviations and informal phrasing. A heuristic that catches one
      style misses the others; the LLM normalises across both.
    - The codebase is already structured-output heavy (Instructor +
      Pydantic validators), so reusing the same path keeps the surface
      area small and the failure modes consistent.
    - The marginal cost is small: ~300 prompt tokens per turn against a
      cheap model (gpt-4o-mini by default).

We still merge defensively in code: ``mentioned_technologies`` is always
unioned with the previous list (deduplicated, case-insensitive), so the
LLM cannot accidentally shrink the list between turns.
"""

from __future__ import annotations

import structlog

from app.services.litellm_wrapper_service import LiteLLMMWrapperService
from app.services.sessions import ProjectMetadata

log = structlog.get_logger()


_EXTRACTOR_SYSTEM_PROMPT = """You extract structured project facts from an
ongoing estimation conversation. You will receive:

1. The current known project metadata, as JSON (may be all-null on turn 1).
2. The latest user transcript for this turn.
3. The estimation summary the assistant just produced.

Return the UPDATED full project metadata. Rules:
- Keep any fact already present unless the user explicitly revises it.
- ``project_name``: the project's short, recognisable name if stated.
- ``assumed_team_size``: integer between 1 and 50 if a team size has been
  mentioned or agreed.
- ``mentioned_technologies`` is a CUMULATIVE list — include every previously
  known item as well as anything new. Use lowercase, canonical names
  (e.g. "react", "postgres", not "ReactJS" or "PostgreSQL").
- ``agreed_scope``: a concise summary (1–3 sentences) of what the parties
  have agreed is in scope so far. Refine — do not append turn by turn.
- ``explicit_constraints`` is a CUMULATIVE list of hard constraints the user
  stated (e.g. "must launch before Q3", "budget capped at 50k", "GDPR
  required"). Keep prior entries; add new ones in the user's own wording.
- ``rejected_options`` is a CUMULATIVE list of approaches the user
  explicitly ruled out (e.g. "no Firebase", "no native mobile app"). These
  must be respected on every future turn.
- If a field cannot be inferred, return null (or [] for lists).
- Never invent facts not present in the conversation."""


def _user_message(
    current: ProjectMetadata,
    transcript: str,
    summary: str,
) -> str:
    return (
        f"Current metadata (JSON):\n{current.model_dump_json()}\n\n"
        f"Latest user transcript:\n{transcript}\n\n"
        f"Latest assistant summary:\n{summary}"
    )


def _merge_unique(old: list[str], new: list[str], *, lowercase: bool) -> list[str]:
    """Union of two lists, preserving order and discarding duplicates.

    Used to guarantee that cumulative metadata lists (technologies,
    constraints, rejections) never shrink between turns even if the LLM
    forgets to echo a prior entry.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*old, *new]:
        if not value:
            continue
        canonical = value.strip()
        if not canonical:
            continue
        if lowercase:
            canonical = canonical.lower()
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(canonical)
    return merged


def extract_project_metadata(
    *,
    current: ProjectMetadata,
    transcript: str,
    summary: str,
    llm_wrapper: LiteLLMMWrapperService,
) -> ProjectMetadata:
    """Run the LLM extractor and return a merged ``ProjectMetadata``.

    If the LLM call fails for any reason we log and return ``current``
    unchanged — losing extraction is recoverable; failing the user's
    estimation request because metadata could not be refreshed is not.
    """
    try:
        updated, meta = llm_wrapper.complete_structured(
            system_prompt=_EXTRACTOR_SYSTEM_PROMPT,
            user_message=_user_message(current, transcript, summary),
            response_model=ProjectMetadata,
            max_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001 — extractor failures must not break the turn
        log.warning(
            "metadata_extractor_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return current

    merged = updated.model_copy(
        update={
            "mentioned_technologies": _merge_unique(
                current.mentioned_technologies,
                updated.mentioned_technologies,
                lowercase=True,
            ),
            "explicit_constraints": _merge_unique(
                current.explicit_constraints,
                updated.explicit_constraints,
                lowercase=False,
            ),
            "rejected_options": _merge_unique(
                current.rejected_options,
                updated.rejected_options,
                lowercase=False,
            ),
        }
    )

    log.info(
        "metadata_extracted",
        project_name=merged.project_name,
        team_size=merged.assumed_team_size,
        tech_count=len(merged.mentioned_technologies),
        constraint_count=len(merged.explicit_constraints),
        rejected_count=len(merged.rejected_options),
        scope_chars=len(merged.agreed_scope or ""),
        latency_ms=meta.get("latency_ms"),
        cost_usd=meta.get("cost_usd"),
    )
    return merged
