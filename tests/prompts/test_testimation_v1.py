import pytest

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation_io import DetailLevel, OutputFormat, ProjectType
from app.schemas.request_io import EstimationRequest


def _request(**overrides) -> EstimationRequest:
    defaults = dict(
        description="We need a landing page with a contact form and HubSpot integration.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )
    return EstimationRequest(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# user prompt — <project_description> block
# ---------------------------------------------------------------------------

class TestUserPromptDescription:
    def test_description_wrapped_in_project_description_tag(self):
        system, user = render_estimation_prompt(_request())
        assert "<project_description>" in user
        assert "</project_description>" in user

    def test_description_content_appears_inside_block(self):
        desc = "Custom ERP for a logistics company with three warehouses."
        system, user = render_estimation_prompt(_request(description=desc))
        start = user.index("<project_description>")
        end = user.index("</project_description>")
        block = user[start:end]
        assert desc in block

    def test_different_descriptions_produce_different_user_prompts(self):
        _, user_a = render_estimation_prompt(_request(description="App A " + "x" * 20))
        _, user_b = render_estimation_prompt(_request(description="App B " + "y" * 20))
        assert user_a != user_b


# ---------------------------------------------------------------------------
# system prompt — output_format rendering
# ---------------------------------------------------------------------------

class TestSystemOutputFormat:
    def test_phases_table_format_emits_confidence_pct(self):
        system, _ = render_estimation_prompt(_request(output_format=OutputFormat.PHASES_TABLE))
        assert "confidence_pct" in system

    def test_phases_table_format_keyword_in_system(self):
        system, _ = render_estimation_prompt(_request(output_format=OutputFormat.PHASES_TABLE))
        assert "phases_table" in system

    def test_narrative_format_does_not_emit_confidence_pct(self):
        system, _ = render_estimation_prompt(_request(output_format=OutputFormat.NARRATIVE))
        assert "confidence_pct" not in system

    def test_narrative_format_keyword_in_system(self):
        system, _ = render_estimation_prompt(_request(output_format=OutputFormat.NARRATIVE))
        assert "narrative" in system


# ---------------------------------------------------------------------------
# system prompt — detail_level rendering
# ---------------------------------------------------------------------------

class TestSystemDetailLevel:
    def test_detailed_includes_assumptions_instruction(self):
        system, _ = render_estimation_prompt(_request(detail_level=DetailLevel.DETAILED))
        assert "assumptions" in system

    def test_detailed_includes_confidence_interval_instruction(self):
        system, _ = render_estimation_prompt(_request(detail_level=DetailLevel.DETAILED))
        assert "confidence" in system

    def test_summary_omits_assumptions_instruction(self):
        system, _ = render_estimation_prompt(_request(detail_level=DetailLevel.SUMMARY))
        assert "assumptions" not in system
