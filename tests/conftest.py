import os

# CRITICAL: set the test API keys *before* importing ``app.main``.
#
# ``app/services/open_ai_service.py`` defines:
#     def __init__(self, client=OpenAI(api_key=get_settings().OPENAI_API_KEY)):
# That default argument is evaluated at *class definition time*, which means
# the call to ``get_settings()`` runs the moment the module is imported. With
# no ``.env`` and no real keys in the environment (as in GitHub CI), the
# ``Settings`` validator raises and conftest collection fails before any test
# is collected.
#
# Setting these env vars before the first ``from app...`` import lets the
# legacy import chain construct without complaining; tests that need real
# keys still skip themselves accordingly.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI test client configured with the application."""
    return TestClient(app)


