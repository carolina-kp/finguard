import pytest
from fastapi.testclient import TestClient

from finguard.api.main import app


def test_health_endpoint_returns_200():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"


