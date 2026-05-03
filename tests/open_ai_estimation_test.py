from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas.llm_io import OpenAIEstimation
from app.services.open_ai_service import OpenAIEstimator


def test_openai_estimator_returns_mocked_estimation(monkeypatch):
    monkeypatch.setattr(settings, "OPEN_API_KEY", "test-api-key", raising=False)
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-4", raising=False)

    fake_response = MagicMock()
    fake_response.output_text = "Mocked estimation text"

    fake_client = MagicMock()
    fake_client.responses.create.return_value = fake_response

    estimator = OpenAIEstimator(client=fake_client)
    result = estimator.estimate("system prompt", "user prompt")

    assert isinstance(result, OpenAIEstimation)
    assert result.estimation == "Mocked estimation text"
    assert result.provider_info.provider == "openai"
    assert result.provider_info.model == "gpt-4"
    assert result.token_usage.input_tokens == 0
    assert result.token_usage.output_tokens == 0
    assert result.token_usage.total_tokens == 0
    assert result.token_usage.cost_usd == 0.0

    fake_client.responses.create.assert_called_once_with(
        model="gpt-4",
        instructions="system prompt",
        input="user prompt",
        temperature=0.2,
        stream=False,
    )


def test_openai_estimator_streams_text_chunks(monkeypatch):
    monkeypatch.setattr(settings, "OPEN_API_KEY", "test-api-key", raising=False)
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-4", raising=False)

    fake_event_1 = MagicMock(type="response.output_text.delta", delta="Hello")
    fake_event_2 = MagicMock(type="response.output_text.delta", delta=" world")
    fake_usage = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
    fake_response = MagicMock(usage=fake_usage)
    fake_event_3 = MagicMock(type="response.completed", response=fake_response)

    fake_client = MagicMock()
    fake_client.responses.create.return_value = iter([fake_event_1, fake_event_2, fake_event_3])

    estimator = OpenAIEstimator(client=fake_client)
    events = list(estimator.estimate_stream("system prompt", "user prompt"))

    assert len(events) == 3
    assert events[0] == {"type": "delta", "text": "Hello"}
    assert events[1] == {"type": "delta", "text": " world"}
    assert events[2]["type"] == "done"
    assert events[2]["estimation"] == "Hello world"
    assert events[2]["provider"] == "openai"
    assert events[2]["model"] == "gpt-4"
    assert events[2]["token_usage"] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
        "cost_usd": 0.00015,
    }
    assert "timestamp" in events[2]
    fake_client.responses.create.assert_called_once_with(
        model="gpt-4",
        instructions="system prompt",
        input="user prompt",
        temperature=0.2,
        stream=True,
    )


def test_streaming_endpoint_returns_chunked_events(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-4", raising=False)

    fake_event_iterator = iter([
        {"type": "delta", "text": "Hello"},
        {"type": "delta", "text": " world"},
        {
            "type": "done",
            "estimation": "Hello world",
            "provider": "openai",
            "model": "gpt-4",
            "timestamp": "2026-05-03T00:00:00",
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
    ])

    monkeypatch.setattr(
        "app.routers.estimations_route.generate_estimation_stream",
        lambda transcription: fake_event_iterator,
        raising=False,
    )

    client = TestClient(app)
    response = client.post("/api/v1/estimate/stream", json={"transcription": "test"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: delta" in response.text
    assert "event: done" in response.text
    assert "Hello" in response.text
    assert "world" in response.text
    assert "\"type\": \"done\"" in response.text
    assert "\"token_usage\"" in response.text
