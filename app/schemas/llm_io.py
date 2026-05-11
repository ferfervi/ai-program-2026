from pydantic import BaseModel

# Interface models LLM 
class LLMServiceError(Exception):
    """Raised when the LLM provider call fails."""

class LLMProviderInfo(BaseModel):
    provider: str
    model: str

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float = 0.0
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0

class LLMEstimation(BaseModel):
    estimation: str
    provider_info: LLMProviderInfo
    token_usage: TokenUsage
    latency_ms: int = 0
    finish_reason: str = "unknown"
    cache_hit: bool = False

# OpenAI estimation result model

class OpenAIEstimation(BaseModel):
    estimation: str
    provider_info: LLMProviderInfo
    token_usage: TokenUsage
    finish_reason: str = "unknown"
    latency_ms: int = 0
    cache_hit: bool = False

# Anthropic estimation result model

class AnthropicEstimation(BaseModel):
    estimation: str
    provider_info: LLMProviderInfo
    token_usage: TokenUsage
    finish_reason: str = "unknown"
    latency_ms: int = 0
    cache_hit: bool = False

