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

- `make serve` - Start the app locally with auto-reload
- `make stop` - Stop the local development server

### Common targets

- `make estimate` - Send a sample estimation request to the endpoint
- `make health` - Check the health endpoint

## Relevant endpoints

- `GET /health` - Service health status
- `POST /api/v1/estimate` - Generate an estimate from the JSON body transcript
- `GET /docs` - Automatic Swagger documentation

## Project structure

- `app/main.py` - FastAPI entrypoint
- `app/routers/estimations.py` - Estimation route and schemas
- `app/services/llm_service.py` - LLM provider request logic
- `app/config.py` - Settings-based configuration
- `app/data/samples/sample_request_transcription.md` - Default sample transcript

## Notes

Make sure the server is running before executing `make estimate`.
