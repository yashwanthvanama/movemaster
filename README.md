# MoveMaster

MoveMaster is an AI engine that predicts the next chess move from prior moves and/or the current position. It includes a training pipeline on public chess datasets, an evaluation harness with a minimax baseline, and a production-ready FastAPI service for inference.

## Overview

- Objective: Given a chess position or sequence of moves, predict the best next move.
- Core components:
  - Data ingestion from Kaggle Chess Evaluations only.
  - Feature extraction using python-chess (bitboards, legal move masks, SAN/uci parsing).
  - Model: lightweight neural network (PyTorch) for move policy; optional value head.
  - Baseline: classic minimax with alpha-beta pruning (depth-limited) using a simple material/positional eval.
  - Training loop with batching, mixed precision, checkpoints.
  - Evaluation: top-1 accuracy over test positions, engine matchups vs minimax, and calibration.
  - Serving: FastAPI endpoints for health, predict, and analyze (minimax) with Docker support.

## Dataset Source (Kaggle Only)

- Kaggle Chess Evaluations: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations (or similar)
  - Contains positions in FEN with engine evaluations and best move labels.
  - Expected columns: FEN (or fen), BestMove (or best_move/move/bestmove). Others ignored.
  - Use FEN -> board via python-chess to create supervised targets.

Data usage notes:
- Verify the Kaggle dataset license and attribution requirements.
- Keep raw downloads outside repo (gitignored). Place under `data/raw/`.

## Project Structure

```
movemaster/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── scripts/
│   └── preprocess.sh            # Preprocess Kaggle CSV → NPZ shards
├── src/movemaster/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py            # FastAPI app
│   ├── data/
│   │   ├── datasets.py          # NPZ shard loader
│   │   └── preprocess.py        # FEN/CSV parsing to tensors
│   ├── models/
│   │   ├── policy.py            # PyTorch model
│   │   └── minimax.py           # Baseline with alpha-beta
│   ├── training/
│   │   ├── train.py             # Training loop
│   │   └── utils.py             # Training utils, metrics, ckpt
│   └── eval/
│       └── evaluate.py          # Metrics & vs minimax
└── tests/
    └── test_api.py
```

## Training Flow

1. Download Kaggle CSV
   - Download the Kaggle Chess Evaluations CSV and place it at `data/raw/chess_evaluations.csv`.
2. Preprocess
   - Parse positions (FEN) with python-chess.
   - For each position, generate features:
     - Bitboard planes (12 piece planes + side to move + castling + en passant).
     - Legal move mask over the fixed UCI move vocabulary.
   - Target: next move index (from BestMove column).
   - Save to `.npz` shards in `data/processed/`.
3. Train
   - PyTorch model (small ConvNet) over planes.
   - Loss: cross-entropy on legal moves only (masking illegal moves during eval/inference).
   - Optim: AdamW, cosine LR optional, mixed precision optional.
   - Checkpoint: best val accuracy and last.
4. Evaluate
   - Top-1 accuracy and top-k on held-out set or random positions agreement with minimax.

## Minimax Baseline

- Implementation: alpha-beta with simple evaluation:
  - Material balance + mobility term.
- Depth: configurable (e.g., 2–4 plies) to keep it fast.
- Used as a sanity check and comparison for learned policy.

## FastAPI Endpoints

- GET `/health` → `{ status: "ok" }`
- POST `/predict` → Input: `{ fen: string, top_k?: int }` Output: `{ moves: [{uci, prob}], legal_moves: [uci] }`
- POST `/analyze` → Input: `{ fen: string, depth?: int }` Output: `{ best_move: uci, score: centipawns }` (minimax)

## How to Run Locally

- Set up environment
  - Python 3.10+
  - Create venv and install dependencies

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Preprocess Kaggle CSV (assumes `data/raw/chess_evaluations.csv` exists)
```
bash scripts/preprocess.sh  # writes data/processed/shard_000.npz
```

- Train (example)
```
python -m src.movemaster.training.train \
  --data-dir data/processed \
  --epochs 2 --batch-size 256 --lr 3e-4 \
  --out models/policy.pt
```

- Evaluate vs minimax
```
python -m src.movemaster.eval.evaluate --num-positions 100 --minimax-depth 3
```

- Run API
```
uvicorn src.movemaster.api.server:app --reload --port 8000
```

- Example curl
```
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

## Docker

- Build
```
docker build -t movemaster:latest .
```
- Run API
```
docker run --rm -p 8000:8000 movemaster:latest
```

## Notes

- This setup uses only Kaggle Chess Evaluations for training.
- The default model is small and CPU-friendly; swap in larger architectures as needed.
