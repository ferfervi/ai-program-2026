from unittest.mock import MagicMock, patch

import pytest

from app.services.litellm_wrapper_service import LiteLLMMWrapperService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get.return_value = None  # default: cache miss
    return cache


@pytest.fixture
def wrapper(mock_cache):
    with patch("app.services.litellm_wrapper_service.Router"):
        svc = LiteLLMMWrapperService(
            openai_api_key="test-openai-key",
            anthropic_api_key="test-anthropic-key",
            primary_model="openai/gpt-4o-mini",
            fallback_model="anthropic/claude-haiku-4-5",
            timeout=30,
            num_retries=2,
            cache=mock_cache,
        )
    return svc


def _make_blocking_response(content="Estimation text", model="gpt-4o-mini",
                            finish_reason="stop", prompt_tokens=100,
                            completion_tokens=200):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    return response


def _make_text_chunk(text, model="gpt-4o-mini"):
    chunk = MagicMock()
    chunk.usage = None
    chunk.model = model
    chunk.choices[0].delta.content = text
    return chunk


def _make_usage_chunk(prompt_tokens, completion_tokens, model="gpt-4o-mini"):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    chunk = MagicMock()
    chunk.usage = usage
    chunk.model = model
    chunk.choices[0].delta.content = None
    return chunk


# ---------------------------------------------------------------------------
# complete() — blocking
# ---------------------------------------------------------------------------

class TestComplete:
    def test_cache_hit_returns_cached_result(self, wrapper, mock_cache):
        cached = {
            "estimation": "Cached result",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "latency_ms": 500,
            "cost_usd": 0.000015,
        }
        mock_cache.get.return_value = cached

        result = wrapper.complete(system_prompt="sys", user_message="user")

        assert result["cache_hit"] is True
        assert result["estimation"] == "Cached result"
        mock_cache.set.assert_not_called()

    def test_cache_miss_calls_llm_and_caches(self, wrapper, mock_cache):
        wrapper._dispatch = MagicMock(return_value=_make_blocking_response())

        result = wrapper.complete(system_prompt="sys", user_message="user")

        assert result["cache_hit"] is False
        assert result["estimation"] == "Estimation text"
        assert result["model"] == "gpt-4o-mini"
        assert result["provider"] == "openai"
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 200
        assert result["usage"]["total_tokens"] == 300
        assert result["finish_reason"] == "stop"
        assert result["cost_usd"] > 0
        mock_cache.set.assert_called_once()

    def test_cost_is_computed_from_token_counts(self, wrapper, mock_cache):
        wrapper._dispatch = MagicMock(
            return_value=_make_blocking_response(prompt_tokens=1_000_000, completion_tokens=0)
        )

        result = wrapper.complete(system_prompt="sys", user_message="user")

        # gpt-4o-mini input rate: 0.15 USD / 1M tokens
        assert result["cost_usd"] == pytest.approx(0.15, rel=1e-3)

    def test_llm_failure_propagates(self, wrapper, mock_cache):
        wrapper._dispatch = MagicMock(side_effect=RuntimeError("API error"))

        with pytest.raises(RuntimeError, match="API error"):
            wrapper.complete(system_prompt="sys", user_message="user")

        mock_cache.set.assert_not_called()


# ---------------------------------------------------------------------------
# complete_stream() — streaming
# ---------------------------------------------------------------------------

class TestCompleteStream:
    def test_cache_hit_yields_text_then_usage(self, wrapper, mock_cache):
        mock_cache.get.return_value = {
            "estimation": "Cached estimation",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            "cost_usd": 0.000075,
        }

        items = list(wrapper.complete_stream(system_prompt="sys", user_message="user"))

        assert items[0] == "Cached estimation"
        usage = items[1]
        assert isinstance(usage, dict)
        assert usage["input_tokens"] == 50
        assert usage["output_tokens"] == 100
        assert usage["total_tokens"] == 150
        assert usage["cost_usd"] == 0.000075
        mock_cache.set.assert_not_called()

    def test_yields_text_chunks_then_usage_dict(self, wrapper, mock_cache):
        chunks = [
            _make_text_chunk("Hello"),
            _make_text_chunk(" world"),
            _make_usage_chunk(prompt_tokens=80, completion_tokens=160),
        ]
        wrapper._dispatch = MagicMock(return_value=iter(chunks))

        items = list(wrapper.complete_stream(system_prompt="sys", user_message="user"))

        text_items = [i for i in items if isinstance(i, str)]
        usage_items = [i for i in items if isinstance(i, dict)]

        assert text_items == ["Hello", " world"]
        assert len(usage_items) == 1

    def test_usage_dict_contains_token_counts(self, wrapper, mock_cache):
        chunks = [
            _make_text_chunk("Result"),
            _make_usage_chunk(prompt_tokens=80, completion_tokens=160),
        ]
        wrapper._dispatch = MagicMock(return_value=iter(chunks))

        items = list(wrapper.complete_stream(system_prompt="sys", user_message="user"))
        usage = next(i for i in items if isinstance(i, dict))

        assert usage["input_tokens"] == 80
        assert usage["output_tokens"] == 160
        assert usage["total_tokens"] == 240
        assert usage["cost_usd"] > 0

    def test_no_usage_chunk_yields_zeros(self, wrapper, mock_cache):
        wrapper._dispatch = MagicMock(return_value=iter([_make_text_chunk("text")]))

        items = list(wrapper.complete_stream(system_prompt="sys", user_message="user"))
        usage = next(i for i in items if isinstance(i, dict))

        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["cost_usd"] == 0.0

    def test_caches_result_after_stream(self, wrapper, mock_cache):
        chunks = [
            _make_text_chunk("Hello"),
            _make_usage_chunk(prompt_tokens=10, completion_tokens=20),
        ]
        wrapper._dispatch = MagicMock(return_value=iter(chunks))

        list(wrapper.complete_stream(system_prompt="sys", user_message="user"))

        mock_cache.set.assert_called_once()
        cached_value = mock_cache.set.call_args[0][1]
        assert cached_value["estimation"] == "Hello"
        assert cached_value["usage"]["input_tokens"] == 10

    def test_stream_failure_propagates(self, wrapper, mock_cache):
        wrapper._dispatch = MagicMock(side_effect=RuntimeError("Stream error"))

        with pytest.raises(RuntimeError, match="Stream error"):
            list(wrapper.complete_stream(system_prompt="sys", user_message="user"))
