import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import api.app as app_module


class FakeClassifier:
    def __init__(self, adapter_path=None):
        self.adapter_path = adapter_path

    def classify(self, text, use_policy=True):
        if not text.strip():
            raise ValueError("text must not be empty")
        return {"label": "INTERNAL", "confidence": 0.9, "policy_override": False}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "DocumentClassifier", FakeClassifier)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_classify_returns_label(client):
    response = client.post("/classify", json={"text": "Standup moves to 10am."})
    assert response.status_code == 200
    assert response.json()["label"] == "INTERNAL"


def test_empty_text_is_rejected(client):
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_whitespace_text_is_a_client_error(client):
    assert client.post("/classify", json={"text": "   "}).status_code == 400


def test_classify_file_extracts_then_classifies(client):
    response = client.post(
        "/classify-file", files={"file": ("notes.txt", b"Standup moves to 10am.", "text/plain")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "INTERNAL"
    assert body["filename"] == "notes.txt"
    assert body["characters"] == len("Standup moves to 10am.")


def test_classify_file_rejects_unsupported_types(client):
    response = client.post("/classify-file", files={"file": ("photo.png", b"\x89PNG", "image/png")})
    assert response.status_code == 415
