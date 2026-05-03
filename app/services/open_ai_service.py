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
