"""Integration tests for the conversational sessions endpoints (Step 7).

Three scenarios, all driven through ``httpx.AsyncClient`` + ``ASGITransport``:

1. Two-turn conversation — verifies ``project_metadata`` is updated between
   turns and that GET /sessions/{id} reflects the change.
2. PDF attachment — verifies the document content actually reaches the
   estimator (and that an estimation field shifts when the file is attached).
3. Eight-turn loop — verifies the sliding-window history sent to the LLM
   never exceeds the configured ``SESSION_MAX_TURNS`` pairs.

LLM calls are stubbed:
- ``EstimationService`` is replaced by a recording fake via FastAPI's
  ``dependency_overrides``.
- ``extract_project_metadata`` is monkeypatched to a deterministic function
  so the tests are hermetic and fast.
- ``extract_text`` (PDF/DOCX parser) is monkeypatched in test #2 so the
  test does not need to ship binary fixtures.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from app.dependencies import get_estimation_service
from app.main import app
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
)
from app.services import sessions as sessions_module
from app.services.sessions import ConversationHistory, ProjectMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Reset the in-memory session registry between tests."""
    sessions_module._SESSIONS.clear()
    yield
    sessions_module._SESSIONS.clear()


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canned_result(total_cost_eur: int = 30_000) -> EstimationResult:
    return EstimationResult(
        summary="Mid-sized B2B SaaS for equipment loans across teams.",
        total_duration_weeks=8,
        total_cost_eur=total_cost_eur,
        confidence_pct=70,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Workshops and scoping."},
            {"name": "Implementation", "duration_weeks": 6,
             "cost_eur": total_cost_eur - 10_000,
             "summary": "Build the core features."},
            {"name": "QA + launch", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Test pass and production rollout."},
        ],
    )


class RecordingFakeService:
    """Captures every ``estimate*`` call and returns a canned (or smart) result.

    Used in place of the real ``EstimationService`` so we can inspect the
    request, attachments, project_metadata and history that the route fed in.
    """

    def __init__(
        self,
        *,
        cost_resolver=lambda request: 30_000,
    ) -> None:
        self.cost_resolver = cost_resolver
        self.estimate_calls: list[dict[str, Any]] = []
        self.attachment_calls: list[dict[str, Any]] = []

    def estimate(
        self,
        request: EstimationRequest,
        project_metadata: ProjectMetadata | None = None,
        history: ConversationHistory | None = None,
    ) -> EstimationResponse:
        self.estimate_calls.append(
            {
                "request": request,
                "metadata": project_metadata,
                "history_len": len(history) if history else 0,
                "history_pairs": history.completed_turns if history else 0,
                "max_turns": history.max_turns if history else None,
            }
        )
        return EstimationResponse(
            result=_canned_result(self.cost_resolver(request)),
            prompt_version="v1",
            cached=False,
        )

    def estimate_with_attachments(
        self,
        request: EstimationRequest,
        attachments: list[tuple[str, bytes]],
        project_metadata: ProjectMetadata | None = None,
        history: ConversationHistory | None = None,
    ) -> EstimationResponse:
        # Mirror the real service's description-augmentation step so the
        # captured ``request.description`` matches what the LLM would see.
        from app.services.estimation_service import (
            ATTACHMENT_SEPARATOR_TEMPLATE,
            extract_text,
        )

        if attachments:
            parts = [request.description]
            for filename, content in attachments:
                parts.append(
                    ATTACHMENT_SEPARATOR_TEMPLATE.format(
                        filename=filename,
                        text=extract_text(filename, content),
                    )
                )
            effective_request = request.model_copy(
                update={"description": "\n".join(parts)}
            )
        else:
            effective_request = request

        self.attachment_calls.append(
            {
                "request": effective_request,
                "attachments": list(attachments),
                "metadata": project_metadata,
                "history_len": len(history) if history else 0,
                "history_pairs": history.completed_turns if history else 0,
                "max_turns": history.max_turns if history else None,
            }
        )
        return self.estimate(
            effective_request,
            project_metadata=project_metadata,
            history=history,
        )


@pytest.fixture
def fake_service():
    svc = RecordingFakeService()
    app.dependency_overrides[get_estimation_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_estimation_service, None)


def _form(transcript: str) -> dict[str, str]:
    return {
        "transcript": transcript,
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }


# ---------------------------------------------------------------------------
# Test 1 — project_metadata is updated between turns
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_project_metadata_updates_between_turns(
    async_client: httpx.AsyncClient,
    fake_service: RecordingFakeService,
    monkeypatch: pytest.MonkeyPatch,
):
    # Stub the extractor: turn 1 establishes the project; turn 2 enriches it.
    seen_transcripts: list[str] = []

    def fake_extractor(*, current, transcript, summary, llm_wrapper):
        seen_transcripts.append(transcript)
        if len(seen_transcripts) == 1:
            return ProjectMetadata(
                project_name="BookFlow",
                assumed_team_size=3,
                mentioned_technologies=["rails"],
            )
        # Turn 2 — should preserve the name + team size, extend technologies.
        return current.model_copy(
            update={
                "mentioned_technologies": ["rails", "postgres"],
                "agreed_scope": "MVP for book management",
                "explicit_constraints": ["must launch before Q3"],
            }
        )

    monkeypatch.setattr(
        "app.routers.sessions_route.extract_project_metadata", fake_extractor
    )

    r = await async_client.post("/api/v1/sessions")
    assert r.status_code == 201
    session_id = r.json()["session_id"]

    # ---- Turn 1 ----
    r1 = await async_client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=_form(
            "Initial brief for BookFlow — a book inventory tool, three engineers."
        ),
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    md1 = body1["project_metadata"]
    assert md1["project_name"] == "BookFlow"
    assert md1["assumed_team_size"] == 3
    assert "rails" in md1["mentioned_technologies"]
    assert body1["completed_turns"] == 1

    state1 = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert state1["project_metadata"]["project_name"] == "BookFlow"
    assert state1["completed_turns"] == 1

    # ---- Turn 2 ----
    r2 = await async_client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=_form(
            "Follow-up: we'll persist to PostgreSQL and need to launch in Q2."
        ),
    )
    assert r2.status_code == 200, r2.text
    md2 = r2.json()["project_metadata"]

    # Name + team size survive; tech list grows; scope/constraints now set.
    assert md2["project_name"] == "BookFlow"
    assert md2["assumed_team_size"] == 3
    assert set(md2["mentioned_technologies"]) >= {"rails", "postgres"}
    assert md2["agreed_scope"] == "MVP for book management"
    assert "must launch before Q3" in md2["explicit_constraints"]

    # Service saw the second turn with the *previous* metadata populated.
    assert len(fake_service.estimate_calls) == 2
    second_call = fake_service.estimate_calls[1]
    assert second_call["metadata"].project_name == "BookFlow"
    assert second_call["history_pairs"] == 1  # one closed pair before turn 2


# ---------------------------------------------------------------------------
# Test 2 — PDF attachment content reaches the LLM layer and changes output
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pdf_attachment_influences_estimation(
    async_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
):
    # Mock the PDF/DOCX parser so we don't need a real binary fixture.
    monkeypatch.setattr(
        "app.services.estimation_service.extract_text",
        lambda filename, content: f"PROJECT_FROM_PDF: BookFlow uses Django and Celery ({filename}, {len(content)} bytes)",
    )

    # Swap in a smart fake that returns a higher cost when the description
    # mentions the project name embedded in the PDF.
    def cost_resolver(request: EstimationRequest) -> int:
        return 80_000 if "BookFlow" in request.description else 20_000

    svc = RecordingFakeService(cost_resolver=cost_resolver)
    app.dependency_overrides[get_estimation_service] = lambda: svc

    # No-op extractor so metadata doesn't leak between the two scenarios.
    monkeypatch.setattr(
        "app.routers.sessions_route.extract_project_metadata",
        lambda *, current, transcript, summary, llm_wrapper: ProjectMetadata(),
    )

    try:
        # ---- Scenario A: no attachment ----
        r_a = await async_client.post("/api/v1/sessions")
        session_a = r_a.json()["session_id"]
        resp_a = await async_client.post(
            f"/api/v1/sessions/{session_a}/estimate",
            data=_form("Plain transcript: generic SaaS for SMB equipment loans."),
        )
        assert resp_a.status_code == 200, resp_a.text
        cost_a = resp_a.json()["estimation"]["result"]["total_cost_eur"]

        # ---- Scenario B: same transcript, with a PDF ----
        r_b = await async_client.post("/api/v1/sessions")
        session_b = r_b.json()["session_id"]
        resp_b = await async_client.post(
            f"/api/v1/sessions/{session_b}/estimate",
            data=_form("Plain transcript: generic SaaS for SMB equipment loans."),
            files=[
                (
                    "attachments",
                    ("brief.pdf", b"%PDF-fake-bytes-content-irrelevant", "application/pdf"),
                )
            ],
        )
        assert resp_b.status_code == 200, resp_b.text
        cost_b = resp_b.json()["estimation"]["result"]["total_cost_eur"]

        # The estimation field actually changes.
        assert cost_a != cost_b, "Attachment did not influence the estimation."

        # Both scenarios route through estimate_with_attachments (the route
        # always calls it, with an empty list when there's no file).
        assert len(svc.attachment_calls) == 2

        no_pdf_call = svc.attachment_calls[0]
        assert no_pdf_call["attachments"] == []
        assert "BookFlow" not in no_pdf_call["request"].description

        with_pdf_call = svc.attachment_calls[1]
        assert [name for name, _ in with_pdf_call["attachments"]] == ["brief.pdf"]
        assert "BookFlow" in with_pdf_call["request"].description
        assert "--- attachment: brief.pdf ---" in with_pdf_call["request"].description
    finally:
        app.dependency_overrides.pop(get_estimation_service, None)


# ---------------------------------------------------------------------------
# Test 3 — 8 turns must not exceed SESSION_MAX_TURNS in the effective history
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_sliding_window_bounds_history_to_max_turns(
    async_client: httpx.AsyncClient,
    fake_service: RecordingFakeService,
    monkeypatch: pytest.MonkeyPatch,
):
    # Hermetic: extractor returns empty metadata so the test focuses on the
    # window logic, not on metadata growth.
    monkeypatch.setattr(
        "app.routers.sessions_route.extract_project_metadata",
        lambda *, current, transcript, summary, llm_wrapper: current,
    )

    from app.config import get_settings

    max_turns = get_settings().SESSION_MAX_TURNS

    r = await async_client.post("/api/v1/sessions")
    session_id = r.json()["session_id"]

    # Send 8 turns. Each turn appends (user, assistant) AFTER the LLM call,
    # so the history-at-call-time is what we want to bound.
    for i in range(8):
        resp = await async_client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data=_form(
                f"Turn {i + 1}: an evolving description of the project — "
                f"please refine the estimate based on what we've discussed."
            ),
        )
        assert resp.status_code == 200, resp.text

    # Pull the inspectable snapshot.
    state = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()

    assert state["max_turns"] == max_turns
    assert state["completed_turns"] <= max_turns
    assert len(state["history"]) <= 2 * max_turns

    # The history sent to the LLM on every turn must never have exceeded
    # the configured window. Each captured history snapshot represents the
    # state BEFORE that turn's transcript was appended, so the bound is
    # ``2 * max_turns`` for completed pairs (max) — strictly: the snapshot
    # never carries more than max_turns completed pairs.
    snapshots = fake_service.estimate_calls
    assert len(snapshots) == 8
    for snap in snapshots:
        assert snap["history_len"] <= 2 * max_turns, (
            f"history_len={snap['history_len']} exceeded "
            f"2 * MAX_TURNS ({2 * max_turns})"
        )
        assert snap["history_pairs"] <= max_turns

    # And the latest turns should be exactly at the cap (we sent 8 ≥ 6).
    assert snapshots[-1]["history_pairs"] == max_turns
