import io
import os
import math
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import chess

# ----------------------
# Move vocabulary
# ----------------------

def _build_move_vocab() -> List[str]:
    voc = []
    squares = list(chess.SQUARES)
    # All from->to (exclude same-square)
    for f in squares:
        for t in squares:
            if f == t:
                continue
            u = chess.SQUARE_NAMES[f] + chess.SQUARE_NAMES[t]
            voc.append(u)
    # Promotions (to q,r,b,n) for white (rank 7 -> 8) and black (rank 2 -> 1)
    promo_pieces = ["q", "r", "b", "n"]
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    # White: from 7th rank
    for f in files:
        from_sq = f + "7"
        # forward, and diagonals
        for to_sq in [f + "8"]:
            for p in promo_pieces:
                voc.append(from_sq + to_sq + p)
        left_file = chr(ord(f) - 1) if f != "a" else None
        right_file = chr(ord(f) + 1) if f != "h" else None
        if left_file:
            for p in promo_pieces:
                voc.append(from_sq + left_file + "8" + p)
        if right_file:
            for p in promo_pieces:
                voc.append(from_sq + right_file + "8" + p)
    # Black: from 2nd rank
    for f in files:
        from_sq = f + "2"
        for to_sq in [f + "1"]:
            for p in promo_pieces:
                voc.append(from_sq + to_sq + p)
        left_file = chr(ord(f) - 1) if f != "a" else None
        right_file = chr(ord(f) + 1) if f != "h" else None
        if left_file:
            for p in promo_pieces:
                voc.append(from_sq + left_file + "1" + p)
        if right_file:
            for p in promo_pieces:
                voc.append(from_sq + right_file + "1" + p)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in voc:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


MOVE_VOCAB: List[str] = _build_move_vocab()
UCI_TO_IDX: Dict[str, int] = {u: i for i, u in enumerate(MOVE_VOCAB)}
VOCAB_SIZE = len(MOVE_VOCAB)


def uci_to_index(uci: str) -> Optional[int]:
    return UCI_TO_IDX.get(uci)


def index_to_uci(idx: int, board: Optional[chess.Board] = None) -> str:
    return MOVE_VOCAB[idx]


def moves_to_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros((VOCAB_SIZE,), dtype=np.float32)
    for m in board.legal_moves:
        u = m.uci()
        idx = uci_to_index(u)
        if idx is not None:
            mask[idx] = 1.0
    return mask


# ----------------------
# Board encoding
# ----------------------

PIECE_PLANES = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

EXTRA_PLANES = {
    "side_to_move": 12,
    "castle_wk": 13,
    "castle_wq": 14,
    "castle_bk": 15,
    "castle_bq": 16,
    "en_passant": 17,
}

N_PLANES = 18


def board_to_features(board: chess.Board) -> np.ndarray:
    x = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    # Pieces
    for (ptype, color), plane in PIECE_PLANES.items():
        for sq in board.pieces(ptype, color):
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            x[plane, r, c] = 1.0
    # Side to move
    x[EXTRA_PLANES["side_to_move"], :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    # Castling
    if board.has_kingside_castling_rights(chess.WHITE):
        x[EXTRA_PLANES["castle_wk"], :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[EXTRA_PLANES["castle_wq"], :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        x[EXTRA_PLANES["castle_bk"], :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[EXTRA_PLANES["castle_bq"], :, :] = 1.0
    # En passant
    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        x[EXTRA_PLANES["en_passant"], r, c] = 1.0
    return x


# ----------------------
# Preprocessing CLI (Kaggle CSV with flexible columns)
# ----------------------

def _append_sample(features: List[np.ndarray], labels: List[int], board: chess.Board, move: chess.Move):
    x = board_to_features(board)
    idx = uci_to_index(move.uci())
    if idx is None:
        return False
    features.append(x)
    labels.append(idx)
    return True


def preprocess_csv(
    csv_path: str,
    out_path: str,
    max_samples: Optional[int] = None,
    fen_col_arg: Optional[str] = None,
    move_col_arg: Optional[str] = None,
    eval_col_arg: Optional[str] = None,
    minimax_depth: int = 0,
) -> int:
    import pandas as pd
    from ..models.minimax import minimax_best_move

    df = pd.read_csv(csv_path)

    # Determine columns
    fen_col = fen_col_arg or next((c for c in ["FEN", "fen", "position"] if c in df.columns), None)
    mv_col = move_col_arg or next((c for c in ["BestMove", "best_move", "move", "bestmove"] if c in df.columns), None)
    ev_col = eval_col_arg or next((c for c in ["Evaluation", "evaluation", "eval", "score"] if c in df.columns), None)

    if fen_col is None:
        raise ValueError("CSV must contain a FEN column (e.g., 'FEN' or 'fen')")

    if mv_col is None and minimax_depth <= 0:
        raise ValueError(
            "No move label column found (BestMove). Provide --move-col or enable label generation with --minimax-depth > 0."
        )

    X: List[np.ndarray] = []
    y: List[int] = []
    count = 0

    for _, row in df.iterrows():
        try:
            board = chess.Board(str(row[fen_col]))
        except Exception:
            continue

        move_obj: Optional[chess.Move] = None
        if mv_col is not None:
            mv_str = str(row[mv_col]).strip()
            try:
                move_obj = chess.Move.from_uci(mv_str)
                if move_obj not in board.legal_moves and len(mv_str) == 4:
                    # Try default queen promotion
                    move_obj = chess.Move.from_uci(mv_str + "q")
            except Exception:
                move_obj = None
        elif minimax_depth > 0:
            mv_obj, _ = minimax_best_move(board, depth=minimax_depth)
            move_obj = mv_obj

        if move_obj is None or move_obj not in board.legal_moves:
            continue

        if _append_sample(X, y, board, move_obj):
            count += 1
        if max_samples and count >= max_samples:
            break

    if count:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, features=np.stack(X, 0), labels=np.array(y, dtype=np.int64))
    return count


def main():
    p = argparse.ArgumentParser(description="Preprocess Kaggle chess CSV to NPZ shards")
    p.add_argument("--csv", type=str, help="Path to Kaggle CSV", required=True)
    p.add_argument("--out", type=str, help="Output NPZ path", default="data/processed/shard_000.npz")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--fen-col", type=str, default=None, help="Column name for FEN (default: auto-detect)")
    p.add_argument("--move-col", type=str, default=None, help="Column name for BestMove (default: auto-detect)")
    p.add_argument("--eval-col", type=str, default=None, help="Column name for evaluation score (optional)")
    p.add_argument("--minimax-depth", type=int, default=0, help="If >0 and BestMove missing, generate labels with minimax at this depth")
    args = p.parse_args()

    total = preprocess_csv(
        csv_path=args.csv,
        out_path=args.out,
        max_samples=args.max_samples,
        fen_col_arg=args.fen_col,
        move_col_arg=args.move_col,
        eval_col_arg=args.eval_col,
        minimax_depth=args.minimax_depth,
    )
    print(f"Wrote {total} samples to {args.out}")


if __name__ == "__main__":
    main()
