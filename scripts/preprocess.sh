#!/usr/bin/env bash
set -euo pipefail

IN_CSV=${1:-data/raw/chessData.csv}
OUT=${2:-data/processed/shard_000.npz}
MAX=${3:-100000}
DEPTH=${4:-2}

# chessData.csv has columns: FEN, Evaluation
# There is no BestMove column; we generate move labels via a shallow minimax.
python -m src.movemaster.data.preprocess \
  --csv "$IN_CSV" \
  --out "$OUT" \
  --max-samples "$MAX" \
  --fen-col FEN \
  --eval-col Evaluation \
  --minimax-depth "$DEPTH"
