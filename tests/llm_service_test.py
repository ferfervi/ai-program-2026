from unittest.mock import MagicMock, patch

import pytest

import app.services.llm_service as llm_service
from app.schemas.llm_io import (
    AnthropicEstimation,
    LLMEstimation,
    LLMProviderInfo,
    OpenAIEstimation,
    TokenUsage,
)
from app.services.llm_service import (
    LLMServiceError,
    build_system_prompt,
    generate_estimation,
    generate_estimation_stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_usage(**kwargs) -> TokenUsage:
    defaults = dict(input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001)
    return TokenUsage(**{**defaults, **kwargs})


def _fake_settings(provider: str = "openai", model: str = "gpt-4o-mini"):
    s = MagicMock()
    s.LLM_PROVIDER = provider
    s.LLM_MODEL = model
    return s


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_contains_role_and_rates(self):
        prompt = build_system_prompt(use_examples=False)
        assert "senior software consultant" in prompt
        assert "62.50 EUR/hour" in prompt
        assert "50 EUR/hour" in prompt

    def test_examples_included_by_default(self):
        prompt = build_system_prompt(num_examples=1)
        assert len(prompt) > len(build_system_prompt(use_examples=False))

    def test_examples_excluded_when_disabled(self):
        prompt_no_ex = build_system_prompt(use_examples=False)
        assert "reference estimations" not in prompt_no_ex

    def test_inline_cleaning_block_included(self):
        prompt = build_system_prompt(use_examples=False, inline_cleaning=True)
        assert "Informal small talk" in prompt

    def test_inline_cleaning_block_excluded_by_default(self):
        prompt = build_system_prompt(use_examples=False)
        assert "Informal small talk" not in prompt


# ---------------------------------------------------------------------------
# generate_estimation — blocking
# ---------------------------------------------------------------------------

class TestGenerateEstimation:
    def test_openai_provider_returns_llm_estimation(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="openai")
        )
        fake = OpenAIEstimation(
            estimation="Estimated",
            provider_info=LLMProviderInfo(provider="openai", model="gpt-4o-mini"),
            token_usage=_token_usage(),
            finish_reason="stop",
        )
        monkeypatch.setattr(
            llm_service.OpenAIEstimator, "estimate", lambda self, s, u: fake
        )

        result = generate_estimation("some transcript")

        assert isinstance(result, LLMEstimation)
        assert result.estimation == "Estimated"
        assert result.provider_info.provider == "openai"
        assert result.finish_reason == "stop"

    def test_anthropic_provider_returns_llm_estimation(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="anthropic")
        )
        fake = AnthropicEstimation(
            estimation="Anthropic result",
            provider_info=LLMProviderInfo(provider="anthropic", model="claude-haiku-4-5"),
            token_usage=_token_usage(),
        )
        monkeypatch.setattr(
            llm_service.AnthropicEstimator, "estimate", lambda self, msgs: fake
        )

        result = generate_estimation("some transcript")

        assert isinstance(result, LLMEstimation)
        assert result.estimation == "Anthropic result"
        assert result.provider_info.provider == "anthropic"

    def test_lite_llm_provider_returns_llm_estimation(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="lite_llm")
        )
        fake_result = {
            "estimation": "LiteLLM result",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "cost_usd": 0.001,
            "cache_hit": False,
        }
        monkeypatch.setattr(llm_service, "_invoke_lite_llm", lambda **kw: fake_result)

        result = generate_estimation("some transcript")

        assert isinstance(result, LLMEstimation)
        assert result.estimation == "LiteLLM result"
        assert result.token_usage.input_tokens == 10
        assert result.token_usage.output_tokens == 20
        assert result.cache_hit is False

    def test_lite_llm_cache_hit_propagated(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="lite_llm")
        )
        fake_result = {
            "estimation": "Cached",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
            "cost_usd": 0.0,
            "cache_hit": True,
        }
        monkeypatch.setattr(llm_service, "_invoke_lite_llm", lambda **kw: fake_result)

        result = generate_estimation("transcript")

        assert result.cache_hit is True

    def test_lite_llm_failure_raises_llm_service_error(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="lite_llm")
        )
        monkeypatch.setattr(
            llm_service, "_invoke_lite_llm", MagicMock(side_effect=RuntimeError("boom"))
        )

        with pytest.raises(LLMServiceError):
            generate_estimation("transcript")

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="unknown_provider")
        )

        with pytest.raises(ValueError):
            generate_estimation("transcript")


# ---------------------------------------------------------------------------
# generate_estimation_stream — streaming
# ---------------------------------------------------------------------------

class TestGenerateEstimationStream:
    def test_openai_yields_delta_and_done_events(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="openai")
        )
        fake_events = [
            {"type": "delta", "text": "Hello"},
            {"type": "delta", "text": " world"},
            {"type": "done", "estimation": "Hello world", "token_usage": {}},
        ]
        monkeypatch.setattr(
            llm_service.OpenAIEstimator, "estimate_stream", lambda self, s, u: iter(fake_events)
        )

        events = list(generate_estimation_stream("transcript"))

        assert events[0] == {"type": "delta", "text": "Hello"}
        assert events[-1]["type"] == "done"

    def test_anthropic_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="anthropic")
        )

        with pytest.raises(ValueError):
            list(generate_estimation_stream("transcript"))

    def test_lite_llm_yields_delta_events_from_text_chunks(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="lite_llm")
        )
        stream_items = [
            "Part one",
            " part two",
            {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150, "cost_usd": 0.01,
             "model": "gpt-4o-mini", "provider": "openai"},
        ]
        monkeypatch.setattr(
            llm_service, "_invoke_lite_llm_stream", lambda **kw: iter(stream_items)
        )

        events = list(generate_estimation_stream("transcript"))

        delta_events = [e for e in events if e["type"] == "delta"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(delta_events) == 2
        assert delta_events[0]["text"] == "Part one"
        assert delta_events[1]["text"] == " part two"
        assert len(done_events) == 1

    def test_lite_llm_done_event_contains_token_usage(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="lite_llm")
        )
        stream_items = [
            "Result",
            {"input_tokens": 80, "output_tokens": 160, "total_tokens": 240, "cost_usd": 0.05,
             "model": "gpt-4o-mini", "provider": "openai"},
        ]
        monkeypatch.setattr(
            llm_service, "_invoke_lite_llm_stream", lambda **kw: iter(stream_items)
        )

        events = list(generate_estimation_stream("transcript"))
        done = next(e for e in events if e["type"] == "done")

        assert done["token_usage"]["input_tokens"] == 80
        assert done["token_usage"]["output_tokens"] == 160
        assert done["token_usage"]["cost_usd"] == 0.05
        assert done["estimation"] == "Result"
        assert "latency_ms" in done

    def test_lite_llm_stream_failure_raises_llm_service_error(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="lite_llm")
        )
        monkeypatch.setattr(
            llm_service,
            "_invoke_lite_llm_stream",
            MagicMock(side_effect=RuntimeError("stream boom")),
        )

        with pytest.raises(LLMServiceError):
            list(generate_estimation_stream("transcript"))

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(
            llm_service, "get_settings", lambda: _fake_settings(provider="unknown_provider")
        )

        with pytest.raises(ValueError):
            list(generate_estimation_stream("transcript"))
