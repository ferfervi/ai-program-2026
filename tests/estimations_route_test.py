from fastapi.testclient import TestClient

from app.main import app
from app.routers import estimations_route
from app.schemas.llm_io import LLMEstimation, LLMProviderInfo, TokenUsage


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


# def test_estimate_valid_request_returns_200(monkeypatch) -> None:
#     fake_estimation = LLMEstimation(
#         estimation="Fake estimation body.",
#         provider_info=LLMProviderInfo(provider="openai", model="gpt-4o-mini"),
#         token_usage=TokenUsage(
#             input_tokens=10,
#             output_tokens=20,
#             total_tokens=30,
#             cost_usd=0.0001,
#         ),
#         latency_ms=42,
#         finish_reason="stop",
#         cache_hit=False,
#     )
#     monkeypatch.setattr(
#         estimations_route, "estimate", lambda llm_input: fake_estimation
#     )

#     client = TestClient(app)
#     response = client.post(
#         "/api/v1/estimate",
#         json={
#             "description": "Mobile app with login, chat and push notifications.",
#             "project_type": "mobile_app",
#             "detail_level": "detailed",
#             "output_format": "phases_table",
#         },
#     )

#     assert response.status_code == 200
#     body = response.json()
#     assert body["text"] == "Fake estimation body."
#     assert body["provider"] == "openai"
#     assert body["model"] == "gpt-4o-mini"
#     assert body["latency_ms"] == 42
#     assert body["cache_hit"] is False
#     assert body["usage"]["total_tokens"] == 30
