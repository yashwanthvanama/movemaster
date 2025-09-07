from fastapi.testclient import TestClient
from src.movemaster.api.app import app

client = TestClient(app)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_next_move():
    r = client.post("/next_move", json={"fen": START_FEN, "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "moves" in data and "legal_moves" in data


def test_mcts_move():
    r = client.post("/mcts_move", json={"fen": START_FEN, "n_simulations": 16})
    assert r.status_code == 200
    data = r.json()
    assert "best_move" in data
