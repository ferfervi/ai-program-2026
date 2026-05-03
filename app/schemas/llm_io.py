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

class LLMEstimation(BaseModel):
    estimation: str
    provider_info: LLMProviderInfo
    token_usage: TokenUsage

# OpenAI estimation result model

class OpenAIEstimation(BaseModel):
    estimation: str
    provider_info: LLMProviderInfo
    token_usage: TokenUsage

# Anthropic estimation result model

class AnthropicEstimation(BaseModel):
    estimation: str
    provider_info: LLMProviderInfo
    token_usage: TokenUsage


