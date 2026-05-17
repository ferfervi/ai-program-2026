import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI test client configured with the application."""
    return TestClient(app)


