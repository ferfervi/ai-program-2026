from unittest.mock import MagicMock

from app.config import settings
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

    fake_client.responses.create.assert_called_once_with(
        model="gpt-4",
        instructions="system prompt",
        input="user prompt",
        temperature=0.2,
    )
