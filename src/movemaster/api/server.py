from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chess
from typing import List, Optional
import torch
import numpy as np

from ..models.policy import PolicyModel, load_model
from ..models.minimax import minimax_best_move

app = FastAPI(title="MoveMaster API", version="0.1.0")


class PredictIn(BaseModel):
    fen: str
    top_k: int = 3


class MoveProb(BaseModel):
    uci: str
    prob: float


class PredictOut(BaseModel):
    moves: List[MoveProb]
    legal_moves: List[str]


class AnalyzeIn(BaseModel):
    fen: str
    depth: int = 3


class AnalyzeOut(BaseModel):
    best_move: str
    score: int


_model: Optional[PolicyModel] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_model():
    global _model
    if _model is None:
        _model = load_model(device=_device)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    _ensure_model()
    try:
        board = chess.Board(body.fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    legal_uci = [m.uci() for m in board.legal_moves]
    if not legal_uci:
        return PredictOut(moves=[], legal_moves=[])

    # Get model probs over legal moves
    with torch.no_grad():
        from ..data.preprocess import board_to_features, moves_to_mask, uci_to_index, index_to_uci
        x = board_to_features(board)
        x = torch.from_numpy(x).unsqueeze(0).to(_device)
        logits = _model(x)  # (1, V)
        logits = logits[0].detach().cpu().numpy()
        mask = moves_to_mask(board)
        logits[mask == 0] = -1e9
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()

    # Top-k legal
    k = max(1, min(body.top_k, len(legal_uci)))
    top_idx = np.argpartition(-probs, k - 1)[:k]
    top_idx = top_idx[np.argsort(-probs[top_idx])]

    moves = []
    for idx in top_idx:
        uci = index_to_uci(int(idx), board)
        if uci in legal_uci:
            moves.append(MoveProb(uci=uci, prob=float(probs[idx])))
    return PredictOut(moves=moves, legal_moves=legal_uci)


@app.post("/analyze", response_model=AnalyzeOut)
def analyze(body: AnalyzeIn):
    try:
        board = chess.Board(body.fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    best_move, score = minimax_best_move(board, depth=body.depth)
    return AnalyzeOut(best_move=best_move.uci() if best_move else "(none)", score=score)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
