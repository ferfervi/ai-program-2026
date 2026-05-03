import structlog
from typing import Iterator

from app.config import settings
from app.services.open_ai_service import  OpenAIEstimator
from app.services.anthropic_service import AnthropicEstimator
from app.context.examples import ESTIMATION_EXAMPLES



log = structlog.get_logger()

MAX_TOKENS = 4000

from app.schemas.llm_io import LLMEstimation, LLMProviderInfo

def build_system_prompt() -> str:
    """Construye el prompt de sistema con el rol y los ejemplos de estimación."""
    instructions = (
        "Eres un estimador de software experto. Genera una estimación detallada y práctica "
        "basada en ejemplos previos y en la transcripción de una nueva reunión. "
        "Usa los ejemplos como referencia del formato y la profundidad esperada. "
        "Tu respuesta debe incluir un desglose de tareas, un total estimado, equipo recomendado "
        "y duración estimada."
    )

    examples = []
    for index, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        summary = example.get("meeting_summary", "").strip()
        estimation = example.get("estimation", "").strip()
        examples.append(
            f"Ejemplo {index}:\nResumen de la reunión: {summary}\nEstimación:\n{estimation}"
        )

    return f"{instructions}\n\n" + "\n\n".join(examples)


def generate_estimation(transcript: str) -> LLMEstimation:
    """Genera una estimación a partir de la transcripción usando el proveedor configurado."""
    system_prompt = build_system_prompt()
    user_prompt = transcript.strip()

    provider = settings.LLM_PROVIDER.lower()

    match provider:
        case "openai":
            openai_estimation = OpenAIEstimator().estimate(system_prompt, user_prompt)

            return LLMEstimation(
                estimation=openai_estimation.estimation,
                provider_info=openai_estimation.provider_info,
                token_usage=openai_estimation.token_usage
            )
        
        case "anthropic":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            anthropic_estimation = AnthropicEstimator().estimate(messages)
            return LLMEstimation(
                estimation=anthropic_estimation.estimation,
                provider_info=anthropic_estimation.provider_info,
                token_usage=anthropic_estimation.token_usage
            )
        
        case _:
            raise ValueError(
                f"Proveedor de LLM desconocido: {settings.LLM_PROVIDER}. Use 'openai' o 'anthropic'."
            )


def generate_estimation_stream(transcript: str) -> Iterator[dict]:
    """Stream OpenAI estimation events, including partial text and final token usage."""
    system_prompt = build_system_prompt()
    user_prompt = transcript.strip()

    provider = settings.LLM_PROVIDER.lower()

    match provider:
        case "openai":
            return OpenAIEstimator().estimate_stream(system_prompt, user_prompt)

        case "anthropic":
            raise ValueError("Streaming sólo está disponible para el proveedor OpenAI.")

        case _:
            raise ValueError(
                f"Proveedor de LLM desconocido: {settings.LLM_PROVIDER}. Use 'openai' o 'anthropic'."
            )
