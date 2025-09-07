import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import chess
import chess.pgn
import torch
import numpy as np

from ..models.policy import load_model, PolicyModel
from ..models.minimax import minimax_best_move
from ..data.preprocess import board_to_features, moves_to_mask, index_to_uci


@dataclass
class ArenaConfig:
    games: int = 10
    minimax_depth: int = 3
    move_limit: int = 200  # half-moves (plies)
    move_time_ms: Optional[int] = None  # (future) per-move time budget, unused for simple depth search
    out_dir: str = "artifacts/arena"


def policy_choose_move(board: chess.Board, model: PolicyModel, device: torch.device) -> chess.Move:
    x = torch.from_numpy(board_to_features(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0].detach().cpu().numpy()
    mask = moves_to_mask(board)
    logits[mask == 0] = -1e9
    best_idx = int(np.argmax(logits))
    uci = index_to_uci(best_idx, board)
    try:
        mv = chess.Move.from_uci(uci)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass
    # Fallback: first legal
    return next(iter(board.legal_moves))


def play_game(game_idx: int, cfg: ArenaConfig, model: PolicyModel, device: torch.device) -> Tuple[str, chess.pgn.Game]:
    """Play one game. Returns (result, pgn_game).
    Result perspective is standard PGN result string ("1-0", "0-1", "1/2-1/2")."""
    board = chess.Board()
    policy_is_white = (game_idx % 2 == 0)

    game = chess.pgn.Game()
    game.headers["Event"] = "MoveMaster Arena"
    game.headers["Round"] = str(game_idx + 1)
    game.headers["White"] = "policy_model" if policy_is_white else "baseline_minimax"
    game.headers["Black"] = "baseline_minimax" if policy_is_white else "policy_model"
    game.headers["Result"] = "*"
    node = game

    ply = 0
    termination_reason = None
    while True:
        if board.is_game_over() or ply >= cfg.move_limit:
            if ply >= cfg.move_limit and not board.is_game_over():
                termination_reason = "MoveLimit"
            break

        side_policy = (board.turn == chess.WHITE and policy_is_white) or (board.turn == chess.BLACK and not policy_is_white)

        if side_policy:
            move = policy_choose_move(board, model, device)
        else:
            # Minimax baseline (depth-limited)
            move, _ = minimax_best_move(board, depth=cfg.minimax_depth)
            if move is None:
                # No legal move -> game over (should be captured by is_game_over next loop)
                break
        board.push(move)
        node = node.add_variation(move)
        ply += 1

    # Determine result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"  # side to move is mated
        termination_reason = "Checkmate"
    elif board.is_stalemate():
        result = "1/2-1/2"
        termination_reason = "Stalemate"
    elif board.can_claim_fifty_moves() or board.is_fifty_moves():
        result = "1/2-1/2"
        termination_reason = "FiftyMoveRule"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        termination_reason = "InsufficientMaterial"
    elif board.can_claim_threefold_repetition() or board.is_repetition():
        result = "1/2-1/2"
        termination_reason = termination_reason or "ThreefoldRepetition"
    else:
        # Move limit draw
        if termination_reason == "MoveLimit":
            result = "1/2-1/2"
        else:
            # Fallback (shouldn't happen)
            result = "1/2-1/2"
            termination_reason = termination_reason or "Unknown"

    game.headers["Result"] = result
    if termination_reason:
        game.headers["Termination"] = termination_reason
    return result, game


def wilson_interval(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def arena(cfg: ArenaConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    policy_wins = 0
    policy_losses = 0
    draws = 0
    pgn_paths: List[str] = []

    for i in range(cfg.games):
        result, pgn_game = play_game(i, cfg, model, device)
        policy_is_white = (i % 2 == 0)
        if result == "1-0":
            if policy_is_white:
                policy_wins += 1
            else:
                policy_losses += 1
        elif result == "0-1":
            if policy_is_white:
                policy_losses += 1
            else:
                policy_wins += 1
        else:
            draws += 1
        out_path = os.path.join(cfg.out_dir, f"game_{i:03d}.pgn")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(pgn_game))
        pgn_paths.append(out_path)
        print(f"Game {i+1}/{cfg.games} result={result} (W/L/D: {policy_wins}/{policy_losses}/{draws})")

    total = cfg.games
    win_rate = (policy_wins + 0.5 * draws) / max(1, total)
    ci_low, ci_high = wilson_interval(win_rate, total)

    report_lines = [
        "# Arena Results",
        f"Games: {total}",
        f"Policy Wins: {policy_wins}",
        f"Policy Losses: {policy_losses}",
        f"Draws: {draws}",
        f"Win Rate (win + 0.5*draw): {win_rate:.4f}",
        f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]",
        "",
        "## Settings",
        f"Minimax Depth: {cfg.minimax_depth}",
        f"Move Limit (plies): {cfg.move_limit}",
        f"Move Time (ms): {cfg.move_time_ms if cfg.move_time_ms else 'N/A'}",
        "",
        "## PGN Files",
    ]
    report_lines.extend([f"- {os.path.basename(p)}" for p in pgn_paths])

    report_path = os.path.join(cfg.out_dir, "arena_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Arena complete. WinRate={win_rate:.4f} 95%CI=({ci_low:.4f},{ci_high:.4f}). Report -> {report_path}")


def main():
    ap = argparse.ArgumentParser(description="Policy vs Minimax Arena")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--minimax-depth", type=int, default=3)
    ap.add_argument("--move-limit", type=int, default=200, help="Max plies before declaring draw")
    ap.add_argument("--move-time-ms", type=int, default=None, help="(Reserved) per-move time budget in ms")
    ap.add_argument("--out-dir", type=str, default="artifacts/arena")
    args = ap.parse_args()

    cfg = ArenaConfig(
        games=args.games,
        minimax_depth=args.minimax_depth,
        move_limit=args.move_limit,
        move_time_ms=args.move_time_ms,
        out_dir=args.out_dir,
    )
    arena(cfg)


if __name__ == "__main__":
    main()
