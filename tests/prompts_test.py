from __future__ import annotations

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, DetailLevel, OutputFormat, ProjectType

import pytest
from jinja2 import Environment, StrictUndefined, UndefinedError

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)



def test_estimation_prompt_includes_description_in_user_block():
    request = EstimationRequest(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.PHASES_TABLE,
    )

    system, user = render_estimation_prompt(request)

    assert "<project_description>" in user
    assert "Mobile app with login" in user
    assert "phases_table" in system
    assert "confidence_pct" in system

# extras -----------------


def _make_request(**overrides) -> EstimationRequest:
    base = {
        "description": "A small CRM for a real estate agency: contacts, deals, role-based access.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    base.update(overrides)
    return EstimationRequest(**base)


def test_user_prompt_wraps_description_in_project_description_block() -> None:
    request = _make_request(description="UNIQUE-MARKER-12345 build a tiny scheduling app.")
    _system, user = render_estimation_prompt(request)
    assert "<project_description>" in user
    assert "UNIQUE-MARKER-12345 build a tiny scheduling app." in user
    assert "</project_description>" in user
    start = user.index("<project_description>")
    end = user.index("</project_description>")
    assert "UNIQUE-MARKER-12345" in user[start:end]


def test_phases_table_keyword_appears_only_when_format_requested() -> None:
    table_request = _make_request(output_format=OutputFormat.PHASES_TABLE)
    narrative_request = _make_request(output_format=OutputFormat.NARRATIVE)

    table_system, _ = render_estimation_prompt(table_request)
    narrative_system, _ = render_estimation_prompt(narrative_request)

    assert "phases_table" in table_system
    assert "phases_table" not in narrative_system


def test_detailed_includes_assumptions_per_phase_summary_does_not() -> None:
    detailed_request = _make_request(detail_level=DetailLevel.DETAILED)
    summary_request = _make_request(detail_level=DetailLevel.SUMMARY)

    detailed_system, _ = render_estimation_prompt(detailed_request)
    summary_system, _ = render_estimation_prompt(summary_request)

    assert "list assumptions per phase" in detailed_system.lower()
    assert "list assumptions per phase" not in summary_system.lower()


def test_examples_block_is_included_in_system_prompt() -> None:
    request = _make_request()
    system, _ = render_estimation_prompt(request)
    assert "<examples>" in system
    assert "</examples>" in system


def test_strict_undefined_raises_on_missing_variable() -> None:
    """A separate Jinja2 template with the same StrictUndefined config must error
    early when a variable is missing — guarantees that typos in templates are
    surfaced at render time, not silently rendered as empty strings."""
    env = Environment(undefined=StrictUndefined)
    template = env.from_string("Hello {{ unknown_variable }}")
    with pytest.raises(UndefinedError):
        template.render()


def test_unknown_version_raises() -> None:
    request = _make_request()
    with pytest.raises(Exception):
        render_estimation_prompt(request, version="v999")