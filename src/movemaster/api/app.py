from __future__ import annotations

import os
import math
import random
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..models.policy import PolicyModel, load_model
from ..models.minimax import evaluate as static_eval
from ..data.preprocess import board_to_features, moves_to_mask, index_to_uci


MODEL_CKPT_PATH = os.environ.get("MOVEMASTER_MODEL", "artifacts/policy_best.pt")

app = FastAPI(title="MoveMaster API", version="0.2.0")


class NextMoveIn(BaseModel):
    fen: str = Field(..., description="FEN string")
    top_k: int = Field(5, ge=1, le=50)


class MoveProb(BaseModel):
    uci: str
    prob: float


class NextMoveOut(BaseModel):
    moves: List[MoveProb]
    legal_moves: List[str]


class MCTSIn(BaseModel):
    fen: str
    n_simulations: int = Field(128, ge=1, le=5000)
    max_rollout_depth: int = Field(24, ge=4, le=200)
    c_puct: float = 1.4


class VisitStat(BaseModel):
    uci: str
    visits: int
    q: float


class MCTSOut(BaseModel):
    best_move: str
    visits: List[VisitStat]


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[PolicyModel] = None
_tokenizer_loaded = False  # Placeholder if a richer tokenizer introduced later


def _lazy_load_model():
    global _model
    if _model is not None:
        return
    ckpt = MODEL_CKPT_PATH if os.path.isfile(MODEL_CKPT_PATH) else None
    _model = load_model(device=_device, ckpt_path=ckpt)


@app.on_event("startup")
def _on_startup():
    # Eager-load model for faster first request; errors are not fatal here.
    try:
        _lazy_load_model()
    except Exception as e:
        print(f"[startup] Warning: failed to load model: {e}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def _predict_distribution(board: chess.Board) -> Tuple[np.ndarray, List[str]]:
    _lazy_load_model()
    assert _model is not None
    x = torch.from_numpy(board_to_features(board)).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(x)[0].detach().cpu().numpy()
    mask = moves_to_mask(board)
    logits[mask == 0] = -1e9
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    legal_uci = [m.uci() for m in board.legal_moves]
    return probs, legal_uci


@app.post("/next_move", response_model=NextMoveOut)
def next_move(body: NextMoveIn):
    try:
        board = chess.Board(body.fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid FEN")
    if board.is_game_over():
        return NextMoveOut(moves=[], legal_moves=[])
    probs, legal = _predict_distribution(board)
    k = min(body.top_k, len(legal))
    # Collect probs for legal moves via index lookup
    entries: List[MoveProb] = []
    # Build mapping idx->uci using index_to_uci (iterate once over all legal mask indices)
    # Simpler: iterate all legal moves and compute their probability
    from ..data.preprocess import uci_to_index
    scores: List[Tuple[str, float]] = []
    for u in legal:
        idx = uci_to_index(u)
        if idx is None:
            continue
        scores.append((u, float(probs[idx])))
    scores.sort(key=lambda x: -x[1])
    for u, p in scores[:k]:
        entries.append(MoveProb(uci=u, prob=p))
    return NextMoveOut(moves=entries, legal_moves=legal)


# ------------------ Simple MCTS ------------------
class MCTSNode:
    __slots__ = ("board", "parent", "move", "children", "N", "W")

    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, move: Optional[chess.Move] = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.N = 0  # visit count
        self.W = 0.0  # total value (from root player's perspective)

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def expanded(self) -> bool:
        return len(self.children) > 0


def _expand(node: MCTSNode):
    if node.board.is_game_over():
        return
    for mv in node.board.legal_moves:
        b2 = node.board.copy(stack=False)
        b2.push(mv)
        node.children.append(MCTSNode(b2, parent=node, move=mv))


def _rollout(board: chess.Board, max_depth: int) -> float:
    # Random playout with static evaluation at end or terminal
    depth = 0
    while depth < max_depth and not board.is_game_over():
        moves = list(board.legal_moves)
        if not moves:
            break
        mv = random.choice(moves)
        board.push(mv)
        depth += 1
    if board.is_checkmate():
        return 1.0 if board.turn == chess.BLACK else -1.0  # side to move just got mated -> previous player wins
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0.0
    # Static eval scaled to [-1,1]
    val_cp = static_eval(board)
    return max(-1.0, min(1.0, val_cp / 1000.0))


def _select(node: MCTSNode, c_puct: float) -> MCTSNode:
    cur = node
    while cur.expanded() and not cur.board.is_game_over():
        total_N = sum(ch.N for ch in cur.children) + 1e-9
        best = None
        best_score = -1e9
        for ch in cur.children:
            U = c_puct * math.sqrt(total_N) / (1 + ch.N)
            score = ch.Q + U
            if score > best_score:
                best_score = score
                best = ch
        cur = best  # type: ignore
    return cur


def run_mcts(root: MCTSNode, n_sim: int, max_rollout_depth: int, c_puct: float):
    for _ in range(n_sim):
        leaf = _select(root, c_puct)
        if not leaf.expanded() and not leaf.board.is_game_over():
            _expand(leaf)
            if leaf.children:
                leaf = random.choice(leaf.children)
        # Rollout
        rollout_board = leaf.board.copy(stack=False)
        value = _rollout(rollout_board, max_rollout_depth)
        # Backprop (from root player's perspective: root.board.turn at start)
        cur = leaf
        root_turn = root.board.turn
        while cur is not None:
            cur.N += 1
            # If cur.board.turn == root_turn, value from root POV is value; else flip sign
            turn_factor = 1 if cur.board.turn == root_turn else -1
            cur.W += value * turn_factor
            cur = cur.parent


@app.post("/mcts_move", response_model=MCTSOut)
def mcts_move(body: MCTSIn):
    try:
        board = chess.Board(body.fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid FEN")
    if board.is_game_over():
        return MCTSOut(best_move="", visits=[])
    root = MCTSNode(board)
    _expand(root)
    if not root.children:
        return MCTSOut(best_move="", visits=[])
    run_mcts(root, body.n_simulations, body.max_rollout_depth, body.c_puct)
    # Select best by visit count
    children = sorted(root.children, key=lambda c: c.N, reverse=True)
    best = children[0]
    visits = [VisitStat(uci=ch.move.uci(), visits=ch.N, q=ch.Q) for ch in children]
    return MCTSOut(best_move=best.move.uci(), visits=visits)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
