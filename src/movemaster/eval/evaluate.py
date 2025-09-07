import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import chess

from ..models.policy import PolicyModel, load_model
from ..models.minimax import minimax_best_move
from ..data.preprocess import board_to_features, moves_to_mask, index_to_uci


def top1_accuracy(model: PolicyModel, boards: List[chess.Board], device: torch.device) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for board in boards:
            x = board_to_features(board)
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            logits = model(x)[0].detach().cpu().numpy()
            mask = moves_to_mask(board)
            logits[mask == 0] = -1e9
            pred = int(np.argmax(logits))
            uci = index_to_uci(pred, board)
            # Treat correctness as: prediction equals engine best move from minimax depth 2
            mv, _ = minimax_best_move(board, depth=2)
            if mv and mv.uci() == uci:
                correct += 1
            total += 1
    return correct / max(1, total)


def eval_vs_minimax(num_positions: int = 100, depth: int = 3) -> Tuple[float, float]:
    # Random legal positions sampled from short random playouts
    rng = random.Random(0)
    boards: List[chess.Board] = []
    for _ in range(num_positions):
        b = chess.Board()
        for _ in range(rng.randint(5, 30)):
            if b.is_game_over():
                break
            mv = rng.choice(list(b.legal_moves))
            b.push(mv)
        boards.append(b)

    # Compare decisions: how often model picks same as deeper minimax
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    agree = 0
    total = 0
    for b in boards:
        with torch.no_grad():
            x = torch.from_numpy(board_to_features(b)).unsqueeze(0).to(device)
            logits = model(x)[0].detach().cpu().numpy()
            mask = moves_to_mask(b)
            logits[mask == 0] = -1e9
            pred_idx = int(np.argmax(logits))
            pred_uci = index_to_uci(pred_idx, b)
        mv_deep, _ = minimax_best_move(b, depth=depth)
        if mv_deep and mv_deep.uci() == pred_uci:
            agree += 1
        total += 1
    return agree / max(1, total), 0.0


def main():
    p = argparse.ArgumentParser(description="Evaluate MoveMaster vs minimax")
    p.add_argument("--data-dir", type=str, default=None, help="Optional processed data dir for position sampling")
    p.add_argument("--num-positions", type=int, default=100)
    p.add_argument("--minimax-depth", type=int, default=3)
    args = p.parse_args()

    acc, elo = eval_vs_minimax(args.num_positions, args.minimax_depth)
    print(f"agreement_with_minimax@{args.minimax_depth} = {acc:.3f}")


if __name__ == "__main__":
    main()
