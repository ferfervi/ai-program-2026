API_URL ?= http://localhost:8000/api/v1/estimate
HEALTH_URL ?= http://localhost:8000/health
TRANSCRIPTION_FILE ?= app/data/samples/sample_request_transcription.md

estimate:
	@python -c 'from pathlib import Path; import json, subprocess; transcription = Path("$(TRANSCRIPTION_FILE)").read_text(encoding="utf-8").strip(); payload = json.dumps({"transcription": transcription}); subprocess.run(["curl", "-sS", "-X", "POST", "$(API_URL)", "-H", "Content-Type: application/json", "-d", payload], check=True)'

health:
	@bash -c 'status=$$(curl -sS -o /dev/null -w "%{http_code}" "$(HEALTH_URL)"); if [ "$$status" = "200" ]; then echo "OK: $$status"; else echo "FAIL: $$status" >&2; exit 1; fi'

serve:
	@uv run uvicorn app.main:app --reload

stop:
	@pkill -f "uvicorn app.main:app" || true
	@pkill -f "uv run uvicorn app.main:app" || true
