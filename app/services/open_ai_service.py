import time

from typing import Iterator

from app.config import settings
from openai import OpenAI

from app.schemas.llm_io import LLMProviderInfo, TokenUsage, OpenAIEstimation

class OpenAIEstimator:
    OPENAI_PRICING = {
        "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        "gpt-4-0613": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        "gpt-4-32k": {"prompt": 0.06 / 1000, "completion": 0.12 / 1000},
        "gpt-3.5-turbo": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
        "gpt-3.5-turbo-16k": {"prompt": 0.003 / 1000, "completion": 0.004 / 1000},
    }

    def __init__(self, client = OpenAI(api_key=settings.OPEN_API_KEY)):
        if not settings.OPEN_API_KEY:
            raise ValueError("OPEN_API_KEY no está configurada.")
        self.client = client
        
    def _parse_usage(self, usage_obj):
        input_tokens = getattr(usage_obj, "input_tokens", None)
        if not isinstance(input_tokens, int):
            input_tokens = getattr(usage_obj, "prompt_tokens", None)
        if not isinstance(input_tokens, int):
            input_tokens = 0

        output_tokens = getattr(usage_obj, "output_tokens", None)
        if not isinstance(output_tokens, int):
            output_tokens = getattr(usage_obj, "completion_tokens", None)
        if not isinstance(output_tokens, int):
            output_tokens = 0

        total_tokens = getattr(usage_obj, "total_tokens", None)
        if not isinstance(total_tokens, int):
            total_tokens = 0

        return input_tokens, output_tokens, total_tokens

    def _find_pricing(self, model_name: str) -> dict:
        if not model_name:
            return {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000}

        normalized = model_name.lower()
        if normalized in self.OPENAI_PRICING:
            return self.OPENAI_PRICING[normalized]

        # Match by prefix so similar model variants still get pricing.
        for key in sorted(self.OPENAI_PRICING.keys(), key=len, reverse=True):
            if normalized.startswith(key):
                return self.OPENAI_PRICING[key]

        return {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000}

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self._find_pricing(settings.LLM_MODEL)
        prompt_rate = pricing.get("prompt", 0.0)
        completion_rate = pricing.get("completion", 0.0)
        return round(input_tokens * prompt_rate + output_tokens * completion_rate, 6)

    def estimate(self,system_prompt: str, user_prompt: str) -> OpenAIEstimation:
        response = self.client.responses.create(
            model=settings.LLM_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.2,
            stream=False
        )

        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            input_tokens, output_tokens, total_tokens = self._parse_usage(usage_obj)
            token_usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=self._estimate_cost(input_tokens, output_tokens),
            )
        else:
            token_usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0)

        output_text = getattr(response, "output_text", None)
        if output_text:
            return OpenAIEstimation(
                estimation=output_text.strip(),
                provider_info=LLMProviderInfo(provider="openai", model=settings.LLM_MODEL),
                token_usage=token_usage,
                finish_reason=getattr(response, "finish_reason", "unknown")
            )

        # Fallback para versiones que exponen la respuesta en output.
        if getattr(response, "output", None):
            output = response.output
            if output and getattr(output[0], "content", None):
                return OpenAIEstimation(
                    estimation=output[0].content[0].text.strip(),
                    provider_info=LLMProviderInfo(provider="openai", model=settings.LLM_MODEL),
                    token_usage=token_usage,
                    finish_reason=getattr(response, "finish_reason", "unknown")
                )

        raise RuntimeError("Respuesta de OpenAI inválida o inesperada.")

    def estimate_stream(self, system_prompt: str, user_prompt: str) -> Iterator[dict]:
        response = self.client.responses.create(
            model=settings.LLM_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.2,
            stream=True,
        )

        t0 = time.perf_counter()

        accumulated = ""
        last_token_usage = None

        for event in response:
            event_type = getattr(event, "type", None)

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    accumulated += delta
                    yield {"type": "delta", "text": delta}

            elif event_type == "response.output_text.done":
                final_text = getattr(event, "text", "")
                if final_text:
                    accumulated += final_text
                    yield {"type": "delta", "text": final_text}

            elif event_type == "response.completed":
                response_obj = getattr(event, "response", None)
                usage_obj = getattr(response_obj, "usage", None)
                if usage_obj is not None:
                    input_tokens, output_tokens, total_tokens = self._parse_usage(usage_obj)
                    cost_usd = self._estimate_cost(input_tokens, output_tokens)
                    last_token_usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "cost_usd": cost_usd,
                    }
                yield {
                    "type": "done",
                    "estimation": accumulated.strip(),
                    "provider": "openai",
                    "model": settings.LLM_MODEL,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "token_usage": last_token_usage or {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                    },
                }
                return

        yield {
            "type": "done",
            "estimation": accumulated.strip(),
            "provider": "openai",
            "model": settings.LLM_MODEL,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "token_usage": last_token_usage or {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            },
        }
