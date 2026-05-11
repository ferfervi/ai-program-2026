from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal


PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]

class StructureCheck(BaseModel):
    """Level-1 structural evaluation of the generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"

class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"

class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"

