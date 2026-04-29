from typing import Dict, List

from anthropic import Anthropic
from openai import OpenAI
from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES


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


def _openai_completion(system_prompt: str, user_prompt: str) -> str:
    if not settings.OPEN_API_KEY:
        raise ValueError("OPEN_API_KEY no está configurada.")

    client = OpenAI(api_key=settings.OPEN_API_KEY)
    response = client.responses.create(
        model=settings.LLM_MODEL,
        instructions=system_prompt,
        input=user_prompt,
        temperature=0.2,
    )

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    # Fallback para versiones que exponen la respuesta en output.
    if getattr(response, "output", None):
        output = response.output
        if output and getattr(output[0], "content", None):
            return output[0].content[0].text.strip()

    raise RuntimeError("Respuesta de OpenAI inválida o inesperada.")


def _anthropic_completion(messages: List[Dict[str, str]]) -> str:
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY no está configurada.")

    anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = anthropic_client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=1024,
        system="Eres un estimador de software experto con 20 años de experiencia. Responde de manera directa y técnica.",
        messages=[
            {"role": message["role"], "content": message["content"].strip()}
            for message in messages
        ],
    )

    if getattr(response, "completion", None):
        return response.completion.strip()

    if getattr(response, "output", None):
        output = response.output
        if output and getattr(output[0], "content", None):
            return output[0].content[0].text.strip()

    raise RuntimeError("Respuesta de Anthropic inválida o inesperada.")


def generate_estimation(transcript: str) -> str:
    """Genera una estimación a partir de la transcripción usando el proveedor configurado."""
    system_prompt = build_system_prompt()
    user_prompt = transcript.strip()

    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        return _openai_completion(system_prompt, user_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if provider == "anthropic":
        return _anthropic_completion(messages)

    raise ValueError(
        f"Proveedor de LLM desconocido: {settings.LLM_PROVIDER}. Use 'openai' o 'anthropic'."
    )
