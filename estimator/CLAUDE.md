# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Development
make server         # Start FastAPI server with auto-reload (port 8000)
make ui             # Start chat Streamlit UI (streamlit_app.py)
make ui-form        # Start form Streamlit UI (streamlit_app_form.py)

# Testing & health
make test           # Run pytest -v
make health         # GET /health
make estimate       # POST /api/v1/estimate with sample description

# Docker
make start-docker   # docker compose up --build (API + Redis)
make stop-docker    # docker compose down
make stop           # Kill local uvicorn process
```

Run a single test: `uv run pytest tests/health_test.py -v`

Dependencies are managed with `uv` (Python 3.11+). The lock file is `uv.lock`.

## Local development (recommended)

```bash
docker-compose up redis   # start Redis (required for caching)
make server               # start FastAPI on :8000
make ui-form              # start form UI on :8501 (recommended)
```

## Environment

Copy `.env.example` to `.env`. Required: at least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Key settings:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `lite_llm` |
| `LLM_MODEL` | `gpt-4o-mini` | model name for provider |
| `PRIMARY_MODEL` | — | LiteLLM primary (e.g. `openai/gpt-4o-mini`) |
| `FALLBACK_MODEL` | — | LiteLLM fallback model |
| `REDIS_URL` | `redis://localhost:6379` | for response caching |
| `CACHE_TTL` | `86400` | seconds |

`docker-compose.yml` runs the API (port 8000) + Redis (port 6379) together.

## Architecture

This is a FastAPI service that takes structured estimation requests and generates software project estimates using LLMs.

### Request body (`EstimationRequest`)

```json
{
  "description": "...",
  "project_type": "web_saas | mobile_app | internal_tool | data_pipeline",
  "detail_level": "summary | medium | detailed",
  "output_format": "phases_table | line_items | narrative"
}
```

### Request flow

```
POST /api/v1/estimate  (or /estimate/stream for SSE)
  → EstimationRequest (description, project_type, detail_level, output_format)
  → render_estimation_prompt()   # Jinja2 → (system_prompt, user_prompt)
  → llm_service.generate_estimation(LLMInputModel)
      → optional preprocessing  # "none" | "inline_cleaning" | "two_phase"
      → route to provider:
          openai_service  /  anthropic_service  /  litellm_wrapper_service
  → EstimationResponse (text, model, provider, usage, cost_usd, latency_ms, cache_hit)
```

### Prompt rendering (`app/prompts/loader.py`)

`render_estimation_prompt(request)` renders Jinja2 templates from `app/prompts/estimation/v1/`:
- `system.j2` — role, output format instructions, detail level instructions, `{% include "examples.j2" %}`
- `user.j2` — wraps `description` in `<project_description>` tags
- `examples.j2` — few-shot estimation examples

### LLM service (`app/services/llm_service.py`)

Central orchestration layer. `generate_estimation(llm_input)` and `generate_estimation_stream(llm_input)` accept an `LLMInputModel(system, user)` and dispatch to the configured provider.

### LiteLLM wrapper (`app/services/litellm_wrapper_service.py`)

Preferred provider path (`LLM_PROVIDER=lite_llm`). Wraps LiteLLM with:
- **Automatic fallback**: `PRIMARY_MODEL` → `FALLBACK_MODEL` on failure
- **Redis cache**: keyed on SHA-256(system_prompt + user_msg + model + max_tokens + thinking_budget); exact-match only
- **Cost tracking**: works for OpenAI and Anthropic models
- **Streaming**: `complete_stream()` yields `str` chunks then a final `dict` with token usage and `cache_hit`

### Provider integrations

- `app/services/open_ai_service.py` — OpenAI SDK, streaming, token/cost calculation
- `app/services/anthropic_service.py` — Anthropic SDK, basic (non-streaming)
- Both are bypassed when `LLM_PROVIDER=lite_llm`

### Schemas (`app/schemas/`)

- `request_io.py` — `EstimationRequest`, `EstimationResponse`
- `estimation_io.py` — `ProjectType`, `DetailLevel`, `OutputFormat` enums
- `llm_io.py` — internal LLM service models (`LLMInputModel`, `TokenUsage`, `LLMEstimation`, …)

### Streaming (SSE)

`POST /api/v1/estimate/stream` returns Server-Sent Events:
- `event: delta` — `{"text": "..."}` partial chunk
- `event: done` — full response JSON including `estimation`, `token_usage`, `latency_ms`, `cache_hit`

Two Streamlit UIs consume this:
- `streamlit_app_form.py` — structured form with streaming toggle (`make ui-form`) **← recommended**
- `streamlit_app.py` — chat interface (`make ui`)

### Caching (`app/services/cache_service.py`)

Redis-backed. The cache key is SHA-256 of the concatenated system prompt, user message, model, max_tokens, and thinking_budget. Cache TTL defaults to 86400 s. Cache hits skip the LLM call entirely and return the stored response with `cache_hit: true`.
