API_URL ?= http://localhost:8000/api/v1/estimate
HEALTH_URL ?= http://localhost:8000/health
TRANSCRIPTION_FILE ?= app/data/samples/sample_request_transcription.md

estimate:
	@python -c 'from pathlib import Path; import json, subprocess; description = Path("$(TRANSCRIPTION_FILE)").read_text(encoding="utf-8").strip(); payload = json.dumps({"description": description, "project_type": "web_saas", "detail_level": "medium", "output_format": "phases_table"}); subprocess.run(["curl", "-sS", "-X", "POST", "$(API_URL)", "-H", "Content-Type: application/json", "-d", payload], check=True)'

estimate-missing-fields:
	@python -c "from pathlib import Path; import json, subprocess, sys; description = Path('$(TRANSCRIPTION_FILE)').read_text(encoding='utf-8').strip(); payload = json.dumps({'description': description, 'detail_level': 'medium', 'output_format': 'phases_table'}); result = subprocess.run(['curl', '-s', '--fail-with-body', '-w', 'HTTP_STATUS:%{http_code}', '-X', 'POST', '$(API_URL)', '-H', 'Content-Type: application/json', '-d', payload], capture_output=True, text=True); print('--- SALIDA DEL SERVIDOR ---'); print(result.stdout.strip()); print('---------------------------'); sys.exit(0 if result.returncode == 0 or result.returncode == 22 else result.returncode)"

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
