API_URL ?= http://localhost:8000/api/v1/estimate
HEALTH_URL ?= http://localhost:8000/health
TRANSCRIPTION_FILE ?= app/data/samples/sample_request_transcription.md

estimate:
	@python -c 'from pathlib import Path; import json, subprocess; description = Path("$(TRANSCRIPTION_FILE)").read_text(encoding="utf-8").strip(); payload = json.dumps({"description": description, "project_type": "web_saas", "detail_level": "medium", "output_format": "phases_table"}); subprocess.run(["curl", "-sS", "-X", "POST", "$(API_URL)", "-H", "Content-Type: application/json", "-d", payload], check=True)'

health:
	@curl -sS "$(HEALTH_URL)"

# Launch with docker
start-docker:
	@docker-compose up --build -d

stop-docker:
	@docker-compose down


# Launch the server in development mode with auto-reload
server:
	@uv run uvicorn app.main:app --reload

ui:
	@streamlit run streamlit_app.py

ui-form:
	@streamlit run streamlit_app_form.py


stop:
	@pkill -f "uvicorn app.main:app" || true
	@pkill -f "uv run uvicorn app.main:app" || true

test:
	@pytest -v 
