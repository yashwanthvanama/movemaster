import argparse
import random
from typing import List, Tuple, Optional, Dict
import json
import math
import os

import numpy as np
import torch
import chess

from ..models.policy import PolicyModel, load_model
from ..models.minimax import minimax_best_move
from ..data.preprocess import board_to_features, moves_to_mask, index_to_uci, uci_to_index


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


def evaluate_jsonl(model: PolicyModel, jsonl_path: str, device: torch.device, topk: int = 3) -> Dict[str, float]:
    """Compute top-1/top-k accuracy and perplexity on a JSONL file.
    JSONL schema: {"fen": str, "next_move": str}
    """
    total = 0
    correct_top1 = 0
    correct_topk = 0
    nll_sum = 0.0

    with torch.no_grad():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    fen = obj["fen"]
                    target_move = obj["next_move"]
                except Exception:
                    continue
                try:
                    board = chess.Board(fen)
                except Exception:
                    continue
                x = torch.from_numpy(board_to_features(board)).unsqueeze(0).to(device)
                logits = model(x)[0]  # (V,)
                mask = moves_to_mask(board)
                logits = logits.detach().cpu().numpy()
                logits[mask == 0] = -1e9
                # Softmax
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                # Target prob
                tgt_idx = uci_to_index(target_move)
                if tgt_idx is None or mask[tgt_idx] == 0:
                    continue
                target_prob = probs[tgt_idx]
                nll_sum += -math.log(max(target_prob, 1e-12))
                total += 1
                # Top-1
                top1 = int(np.argmax(probs))
                if top1 == tgt_idx:
                    correct_top1 += 1
                # Top-k
                k = min(topk, int(mask.sum()))
                if k > 1:
                    topk_idx = np.argpartition(-probs, k - 1)[:k]
                    if tgt_idx in topk_idx:
                        correct_topk += 1
                else:
                    if top1 == tgt_idx:
                        correct_topk += 1
    if total == 0:
        return {"top1": 0.0, "topk": 0.0, "perplexity": float('inf'), "count": 0}
    top1_acc = correct_top1 / total
    topk_acc = correct_topk / total
    ppl = math.exp(nll_sum / total)
    return {"top1": top1_acc, "topk": topk_acc, "perplexity": ppl, "count": total}


def write_markdown_report(metrics: Dict[str, float], out_path: str, topk: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"Total Samples: {int(metrics['count'])}\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Top-1 Accuracy | {metrics['top1']:.4f} |\n")
        f.write(f"| Top-{topk} Accuracy | {metrics['topk']:.4f} |\n")
        f.write(f"| Perplexity | {metrics['perplexity']:.4f} |\n")


def main():
    p = argparse.ArgumentParser(description="Evaluate MoveMaster")
    p.add_argument("--jsonl", type=str, help="Path to test JSONL (fen,next_move)")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--report", type=str, default="artifacts/eval_report.md")
    p.add_argument("--minimax-depth", type=int, default=0, help="Optional: also compute agreement with minimax")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    metrics = evaluate_jsonl(model, args.jsonl, device, topk=args.topk)
    print(f"top1={metrics['top1']:.4f} top{args.topk}={metrics['topk']:.4f} ppl={metrics['perplexity']:.4f} n={metrics['count']}")

    if args.minimax_depth > 0:
        # Optionally compute minimax agreement on same dataset (ignoring labels) just for diagnostics
        agree = 0
        total = 0
        with open(args.jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    fen = obj["fen"]
                    board = chess.Board(fen)
                except Exception:
                    continue
                with torch.no_grad():
                    x = torch.from_numpy(board_to_features(board)).unsqueeze(0).to(device)
                    logits = model(x)[0].detach().cpu().numpy()
                    mask = moves_to_mask(board)
                    logits[mask == 0] = -1e9
                    pred_idx = int(np.argmax(logits))
                mv, _ = minimax_best_move(board, depth=args.minimax_depth)
                if mv and pred_idx == uci_to_index(mv.uci()):
                    agree += 1
                total += 1
        if total:
            metrics[f"minimax_depth_{args.minimax_depth}_agreement"] = agree / total

    write_markdown_report(metrics, args.report, args.topk)
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
