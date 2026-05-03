from datetime import datetime
from typing import Iterator

from app.config import settings
from openai import OpenAI

from app.schemas.llm_io import LLMProviderInfo, TokenUsage, OpenAIEstimation

class OpenAIEstimator:
    def __init__(self, client = OpenAI(api_key=settings.OPEN_API_KEY)):
        if not settings.OPEN_API_KEY:
            raise ValueError("OPEN_API_KEY no está configurada.")
        self.client = client
        
    def estimate(self,system_prompt: str, user_prompt: str) -> OpenAIEstimation:
        response = self.client.responses.create(
            model=settings.LLM_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            temperature=0.2,
            stream=False
        )

        output_text = getattr(response, "output_text", None)
        if output_text:
            return OpenAIEstimation(
                estimation=output_text.strip(),
                provider_info=LLMProviderInfo(provider="openai", model=settings.LLM_MODEL),
                token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
            )

        # Fallback para versiones que exponen la respuesta en output.
        if getattr(response, "output", None):
            output = response.output
            if output and getattr(output[0], "content", None):
                return OpenAIEstimation(
                    estimation=output[0].content[0].text.strip(),
                    provider_info=LLMProviderInfo(provider="openai", model=settings.LLM_MODEL),
                    token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
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
                    last_token_usage = {
                        "input_tokens": getattr(usage_obj, "input_tokens", 0),
                        "output_tokens": getattr(usage_obj, "output_tokens", 0),
                        "total_tokens": getattr(usage_obj, "total_tokens", 0),
                    }
                yield {
                    "type": "done",
                    "estimation": accumulated.strip(),
                    "provider": "openai",
                    "model": settings.LLM_MODEL,
                    "timestamp": datetime.utcnow().isoformat(),
                    "token_usage": last_token_usage or {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                }
                return

        yield {
            "type": "done",
            "estimation": accumulated.strip(),
            "provider": "openai",
            "model": settings.LLM_MODEL,
            "timestamp": datetime.utcnow().isoformat(),
            "token_usage": last_token_usage or {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }
