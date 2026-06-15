from types import SimpleNamespace
from unittest.mock import patch

import fakeredis
import pytest

from app.services.cache_service import EstimationCache
from app.services.litellm_wrapper_service import LiteLLMMWrapperService, _estimate_cost


def _fake_completion(model: str, content: str = "the answer", input_tokens: int = 100, output_tokens: int = 50):
    """Build a SimpleNamespace shaped like a litellm.ModelResponse."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


@pytest.fixture
def wrapper() -> LiteLLMMWrapperService:
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    return LiteLLMMWrapperService(
        openai_api_key="fake-openai",
        anthropic_api_key="fake-anthropic",
        primary_model="gpt-4o-mini",
        fallback_model="claude-haiku-4-5-20251001",
        timeout=30,
        num_retries=2,
        cache=cache,
    )


def test_estimate_cost_uses_pricing_table() -> None:
    cost = _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    # 1M input * 0.15 + 1M output * 0.60 = 0.75 USD
    assert cost == pytest.approx(0.75)


def test_complete_returns_normalised_dict_and_caches(wrapper: LiteLLMMWrapperService) -> None:
    fake = _fake_completion(model="gpt-4o-mini", content="hello world")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked.call_count == 1
    assert result["estimation"] == "hello world"
    assert result["model"] == "gpt-4o-mini"
    assert result["provider"] == "openai"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 50
    assert result["cache_hit"] is False
    assert result["cost_usd"] > 0

    # Second call with the same inputs should hit the cache without invoking the router.
    with patch.object(wrapper.router, "completion") as mocked_again:
        cached = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=None,
        )
    assert mocked_again.call_count == 0
    assert cached["cache_hit"] is True
    assert cached["estimation"] == "hello world"


def test_complete_with_model_override_bypasses_router(wrapper: LiteLLMMWrapperService) -> None:
    fake = _fake_completion(model="gpt-4o", content="overridden")
    with patch("app.services.litellm_wrapper_service.litellm.completion", return_value=fake) as direct, \
        patch.object(wrapper.router, "completion") as router_call:
        result = wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="gpt-4o",
            max_tokens=4000,
            thinking_budget=None,
        )
    assert direct.call_count == 1
    assert router_call.call_count == 0
    assert direct.call_args.kwargs["model"] == "gpt-4o"
    assert result["model"] == "gpt-4o"


def test_thinking_budget_passed_for_anthropic_fallback(wrapper: LiteLLMMWrapperService) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch.object(wrapper.router, "completion", return_value=fake) as mocked:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override=None,
            max_tokens=4000,
            thinking_budget=2048,
        )
    # primary is OpenAI (gpt-4o-mini), so thinking budget is *ignored* in kwargs.
    assert "thinking" not in mocked.call_args.kwargs


def test_thinking_budget_pads_max_tokens_when_anthropic_override(wrapper: LiteLLMMWrapperService) -> None:
    fake = _fake_completion(model="claude-haiku-4-5-20251001", content="ok")
    with patch("app.services.litellm_wrapper_service.litellm.completion", return_value=fake) as direct:
        wrapper.complete(
            system_prompt="sys",
            user_message="usr",
            model_override="claude-haiku-4-5-20251001",
            max_tokens=1000,
            thinking_budget=4096,
        )
    kwargs = direct.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert kwargs["max_tokens"] == 4096 + 1024


# test_complete_stream_yields_chunks_and_caches was removed in Session 4 when
# the /api/v1/estimate/stream endpoint and the wrapper's complete_stream() method
# were deleted. Structured output via Instructor (complete_structured) replaces
# token streaming; tests for that path live in test_estimate_endpoint.py with a
# mocked EstimationService.