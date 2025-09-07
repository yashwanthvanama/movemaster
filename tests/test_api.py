import json
from fastapi.testclient import TestClient
from src.movemaster.api.server import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_startpos():
    client = TestClient(app)
    payload = {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "moves" in data
    assert "legal_moves" in data


def test_analyze_startpos():
    client = TestClient(app)
    payload = {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 2}
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "best_move" in data
    assert "score" in data
