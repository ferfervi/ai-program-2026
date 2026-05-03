
from app.config import settings
from anthropic import Anthropic
from typing import Dict, List

from app.schemas.llm_io import  LLMProviderInfo, TokenUsage, AnthropicEstimation

class AnthropicEstimator:
    def __init__(self, client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)):
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY no está configurada.")
        self.client = client

    def estimate(self, messages: List[Dict[str, str]]) -> AnthropicEstimation:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY no está configurada.")

        response = self.client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=1024,
            system="Eres un estimador de software experto con 20 años de experiencia. Responde de manera directa y técnica.",
            messages=[
                {"role": message["role"], "content": message["content"].strip()}
                for message in messages
            ],
        )

        if getattr(response, "completion", None):
            return AnthropicEstimation(
                estimation=response.completion.strip(),
                provider_info=LLMProviderInfo(provider="anthropic", model=settings.LLM_MODEL),
                token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0),
            )

        if getattr(response, "output", None):
            output = response.output
            if output and getattr(output[0], "content", None):
                return AnthropicEstimation(
                    estimation=output[0].content[0].text.strip(),
                    provider_info=LLMProviderInfo(provider="anthropic", model=settings.LLM_MODEL),
                    token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0),
                )

        raise RuntimeError("Respuesta de Anthropic inválida o inesperada.")
