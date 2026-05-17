"""Endpoint-level tests for POST /api/v1/estimate.

These tests run hermetically: the LLM is never called because the test
overrides ``get_estimation_service`` with a fake that returns a canned
``EstimationResponse``. The fake API keys set in ``conftest.py`` are
enough to satisfy module-level imports (some legacy paths construct an
OpenAI client at class-definition time) — no real provider key required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_estimation_service
from app.main import app
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
)


def _canned_result() -> EstimationResult:
    return EstimationResult(
        summary="Mobile app with login, chat and push — mid-sized scope.",
        total_duration_weeks=8,
        total_cost_eur=30_000,
        confidence_pct=70,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Workshops and scoping."},
            {"name": "Implementation", "duration_weeks": 6, "cost_eur": 20_000,
             "summary": "Build the core features."},
            {"name": "QA + launch", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Test pass and production rollout."},
        ],
    )


class _FakeEstimationService:
    """Records the request and returns a canned response. No LLM call."""

    def __init__(self) -> None:
        self.calls: list[EstimationRequest] = []

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        self.calls.append(request)
        return EstimationResponse(
            result=_canned_result(),
            prompt_version="v1",
            cached=False,
            model="gpt-4o-mini",
            provider="openai",
            latency_ms=42,
            cost_usd=0.0001,
        )


@pytest.fixture
def fake_service():
    svc = _FakeEstimationService()
    app.dependency_overrides[get_estimation_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_estimation_service, None)


def test_estimate_missing_description_returns_422() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/estimate",
        json={
            "project_type": "mobile_app",
            "detail_level": "detailed",
            "output_format": "phases_table",
        },
    )
    assert response.status_code == 422
    assert any(err["loc"][-1] == "description" for err in response.json()["detail"])


def test_estimate_valid_request_returns_200(fake_service: _FakeEstimationService) -> None:
    """200-path: route forwards the request to ``EstimationService`` via DI
    and serialises the structured response back to the caller.

    The fake service is injected through ``app.dependency_overrides`` so
    the real ``LiteLLMMWrapperService`` (and hence any provider HTTP call)
    is never constructed — the test keys from ``conftest.py`` are only
    needed because some legacy modules build an ``OpenAI`` client at
    class-definition time on import.
    """
    client = TestClient(app)
    response = client.post(
        "/api/v1/estimate",
        json={
            "description": "Mobile app with login, chat and push notifications.",
            "project_type": "mobile_app",
            "detail_level": "detailed",
            "output_format": "phases_table",
        },
    )
    assert response.status_code == 200
    body = response.json()

    # Modern response shape: ``result`` (structured) — no ``text`` field.
    assert body["prompt_version"] == "v1"
    assert body["cached"] is False
    assert body["model"] == "gpt-4o-mini"
    assert body["provider"] == "openai"
    assert body["latency_ms"] == 42
    assert body["cost_usd"] == 0.0001
    assert body["result"]["total_cost_eur"] == 30_000
    assert body["result"]["confidence_pct"] == 70
    assert len(body["result"]["phases"]) == 3
    assert body["result"]["phases"][0]["name"] == "Discovery"

    # The route delegated to the service exactly once with the typed payload.
    assert len(fake_service.calls) == 1
    received = fake_service.calls[0]
    assert received.description == "Mobile app with login, chat and push notifications."
    assert received.project_type.value == "mobile_app"
    assert received.detail_level.value == "detailed"
    assert received.output_format.value == "phases_table"
