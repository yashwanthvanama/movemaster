#!/usr/bin/env bash
set -euo pipefail

# This project is configured to use Kaggle Chess Evaluations only.
# Please download the Kaggle CSV (e.g., chess-evaluations.csv) and place it under data/raw/.
# Then run:
#   bash scripts/preprocess.sh data/raw/chess_evaluations.csv data/processed/shard_000.npz

echo "Please download the Kaggle Chess Evaluations CSV to data/raw/ and run scripts/preprocess.sh."
