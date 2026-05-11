# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Development
make serve          # Start FastAPI server with auto-reload (port 8000)
make ui             # Start Streamlit UI (streamlit_app.py)

# Testing & health
make test           # Run pytest -v
make health         # GET /health
make estimate       # POST /api/v1/estimate with sample transcription

# Docker
make start-docker   # docker compose up --build
make stop-docker    # docker compose down
make stop           # Kill local uvicorn process
```

Run a single test: `uv run pytest tests/health_test.py -v`

Dependencies are managed with `uv` (Python 3.11+). The lock file is `uv.lock`.

## Environment

Copy `.env.example` to `.env`. Required: at least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Key settings:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `litellm` |
| `LLM_MODEL` | `gpt-4o-mini` | model name for provider |
| `PRIMARY_MODEL` | — | LiteLLM primary (e.g. `openai/gpt-4o-mini`) |
| `FALLBACK_MODEL` | — | LiteLLM fallback model |
| `REDIS_URL` | `redis://localhost:6379` | for response caching |
| `CACHE_TTL` | `86400` | seconds |

`docker-compose.yml` runs the API (port 8000) + Redis (port 6379) together.

## Architecture

This is a FastAPI service that takes meeting transcriptions and generates software project estimates using LLMs.

### Request flow

```
POST /api/v1/estimate  (or /estimate/stream for SSE)
  → EstimationRequest (transcription, optional thread_id)
  → llm_service.generate_estimation()
      → build_system_prompt()   # role + rates + few-shot examples
      → optional preprocessing  # "none" | "inline_cleaning" | "two_phase"
      → route to provider:
          openai_service  /  anthropic_service  /  litellm_wrapper_service
  → EstimationResponse (estimation text, model, provider, token usage, cost, cache_hit)
```

### LLM service (`app/services/llm_service.py`)

Central orchestration layer. `generate_estimation()` and `generate_estimation_stream()` build the system prompt (assembling role definition, hourly rates — 62.50 EUR/hr dev, 50 EUR/hr design — and few-shot examples from `app/context/examples.py`), optionally pre-processes the transcription, then dispatches to the chosen provider.

### LiteLLM wrapper (`app/services/litellm_wrapper_service.py`)

Preferred provider path. Wraps LiteLLM with:
- **Automatic fallback**: `PRIMARY_MODEL` → `FALLBACK_MODEL` on failure
- **Redis cache**: keyed on SHA-256(system_prompt + user_msg + model + max_tokens + thinking_budget); exact-match only
- **Cost tracking**: works for OpenAI and Anthropic models

### Provider integrations

- `app/services/open_ai_service.py` — OpenAI SDK, streaming, token/cost calculation
- `app/services/anthropic_service.py` — Anthropic SDK, basic (non-streaming)
- Both are bypassed when `LLM_PROVIDER=litellm`

### Schemas (`app/schemas/`)

- `request_io.py` — `EstimationRequest`, `EstimationResponse`
- `estimation_io.py` — estimation-specific Pydantic models
- `llm_io.py` — internal LLM service models (token usage, cost, cache metadata)

### Streaming (SSE)

`POST /api/v1/estimate/stream` returns Server-Sent Events:
- `event: delta` — `{"text": "..."}` partial chunk
- `event: done` — full `EstimationResponse` JSON with token/cost summary

The Streamlit UI (`streamlit_app.py`) consumes this stream and renders chunks in real time. Configure the API target via `API_URL` env var.

### Caching (`app/services/cache_service.py`)

Redis-backed. The cache key is SHA-256 of the concatenated system prompt, user message, model, max_tokens, and thinking_budget. Cache TTL defaults to 86400 s. Cache hits skip the LLM call entirely and return the stored response with `cache_hit: true`.
