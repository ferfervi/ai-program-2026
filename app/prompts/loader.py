"""Jinja2 loader for versioned prompt templates.

The on-disk layout is ``app/prompts/<use_case>/<version>/<role>.j2``. Versioning
is required from day one: switching prompts becomes a string change at the
call site (``version="v2"``), not a code refactor.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest
from app.services.sessions import ProjectMetadata

_BASE_DIR = Path(__file__).resolve().parent

_env = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
    keep_trailing_newline=True,
)


def _metadata_context(metadata: ProjectMetadata | None) -> dict[str, object]:
    """Flatten ``ProjectMetadata`` into the template-friendly shape.

    ``has_facts`` lets the template render an empty <project_metadata>
    block on the first turn without an explicit None-check per field.
    """
    if metadata is None:
        return {
            "project_metadata": {
                "has_facts": False,
                "project_name": None,
                "assumed_team_size": None,
                "mentioned_technologies": [],
                "agreed_scope": None,
                "explicit_constraints": [],
                "rejected_options": [],
            }
        }
    techs = metadata.mentioned_technologies or []
    constraints = metadata.explicit_constraints or []
    rejected = metadata.rejected_options or []
    has_facts = bool(
        metadata.project_name
        or metadata.assumed_team_size
        or techs
        or metadata.agreed_scope
        or constraints
        or rejected
    )
    return {
        "project_metadata": {
            "has_facts": has_facts,
            "project_name": metadata.project_name,
            "assumed_team_size": metadata.assumed_team_size,
            "mentioned_technologies": techs,
            "agreed_scope": metadata.agreed_scope,
            "explicit_constraints": constraints,
            "rejected_options": rejected,
        }
    }


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
    project_metadata: ProjectMetadata | None = None,
) -> tuple[str, str]:
    """Render the system and user prompts for the estimation use case.

    When ``project_metadata`` is provided, its known facts are injected into
    a ``<project_metadata>`` block in the system prompt so the LLM can build
    on what previous turns already established.

    Returns:
        A tuple ``(system_prompt, user_prompt)`` ready to be sent to the LLM
        as separate ``role: "system"`` and ``role: "user"`` messages.
    """
    context = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        **_metadata_context(project_metadata),
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user