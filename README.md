# estimator-cag

Estimator CAG is a FastAPI service for generating software project estimates from meeting transcripts.

## Launch the project

### Execution with Docker

Use Docker Compose to build and start the service in a container.

```bash
make start-docker
```

This runs:

```bash
docker-compose up --build -d
```

To stop the Docker stack:

```bash
make stop-docker
```

### Local execution without Docker

Run the app locally with the project's Python tooling.

```bash
make serve
```

This runs:

```bash
uv run uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`.

## Health check

```bash
make health
```

This sends a request to `http://localhost:8000/health` and verifies that it returns `200`.

## Run a sample estimation request

```bash
make estimate
```

This sends a `POST` request to `http://localhost:8000/api/v1/estimate` with the transcript loaded from:

- `app/data/samples/sample_request_transcription.md`

If you want to use a different transcript file, pass the path with `TRANSCRIPTION_FILE`:

```bash
make estimate TRANSCRIPTION_FILE=app/data/another_transcription.md
```

## Available Make targets

### Docker targets

- `make start-docker` - Build and start the service with Docker Compose
- `make stop-docker` - Stop the Compose services

### Local targets

- `make serve` - Start the FastAPI backend server with auto-reload
- `make ui` - Start the Streamlit UI application
- `make test` - Run the test suite

### Common targets

- `make estimate` - Send a sample estimation request to the endpoint
- `make health` - Check the health endpoint

## Relevant endpoints

- `GET /health` - Service health status
- `POST /api/v1/estimate` - Generate an estimate from the JSON body transcript (full response)
- `POST /api/v1/estimate/stream` - Generate an estimate with server-sent events (SSE) streaming
- `GET /docs` - Automatic Swagger documentation

## Streaming responses

The `/api/v1/estimate/stream` endpoint returns estimation responses using **Server-Sent Events (SSE)** format. This enables real-time streaming of the generated text as it is produced by the LLM.

### How it works

1. Client sends a `POST` request to `/api/v1/estimate/stream` with JSON body: `{"transcription": "...", "thread_id": "..."}`
2. Server starts streaming events as they arrive from the LLM provider
3. Each event has `event:` type and `data:` payload (JSON)

### Event types

- `event: delta` - Partial text chunk (streaming in progress)
  - `text`: A portion of the generated estimation
- `event: done` - Final completion event with metadata
  - `estimation`: Full accumulated estimation text
  - `model`: Model name used
  - `provider`: LLM provider ("openai", "anthropic")
  - `timestamp`: ISO-8601 timestamp
  - `token_usage`: Object with `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`

### Example usage with curl

```bash
curl -X POST http://localhost:8000/api/v1/estimate/stream \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -d '{"transcription":"Build a landing page with contact form", "thread_id":"abc-123"}'
```

### Streamlit UI integration

The Streamlit application (`streamlit_app.py`) consumes this streaming endpoint to:
- Display real-time text generation in the chat interface
- Show a live progress indicator while the LLM generates the response
- Capture token usage and cost metrics upon completion
- Display debug metadata in an expandable info panel

## Project structure

- `streamlit_app.py` - Interactive Streamlit UI for project estimation with real-time SSE streaming
- `app/main.py` - FastAPI entrypoint
- `app/routers/estimations_route.py` - Estimation routes for both standard and streaming endpoints
- `app/services/llm_service.py` - LLM provider request logic and prompt building
- `app/services/open_ai_service.py` - OpenAI API integration with streaming support and cost estimation
- `app/services/anthropic_service.py` - Anthropic API integration
- `app/config.py` - Settings-based configuration
- `app/schemas/llm_io.py` - Request/response schemas including token usage and cost
- `app/data/samples/sample_request_transcription.md` - Default sample transcript

## Streamlit UI

The Streamlit application provides an interactive chat interface for generating project estimates in real-time.

### Running the UI

```bash
make ui
```

This starts the Streamlit app at `http://localhost:8501`.

### Features

- **Real-time streaming**: See estimations appear character-by-character as the LLM generates them
- **Token metrics**: Display input/output token counts and estimated cost for each estimation
- **Session management**: Thread ID tracking and conversation history
- **System prompt visualization**: View the active system prompt and injected context examples
- **Debug info**: Expandable metadata panel showing model, provider, and token usage details
- **Sidebar controls**: Clear chat, start new session, and test backend connectivity

### Configuration

The UI connects to the FastAPI backend at `http://localhost:8000` by default. Override with the `API_URL` environment variable:

```bash
API_URL=http://custom-api:8000 streamlit run streamlit_app.py
```

## Notes

Make sure the FastAPI backend server is running before executing estimation requests or starting the Streamlit UI:

```bash
make server
```
