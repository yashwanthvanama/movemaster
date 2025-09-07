# MoveMaster

An AI engine for predicting the next chess move from a position (FEN) with a lightweight policy network, evaluation utilities, a minimax + MCTS baseline, and a production FastAPI service.

## Highlights
- Kaggle-only supervised data (FEN + evaluation score) with optional move label generation via minimax.
- Feature encoder: 18-plane bitboard representation (12 piece planes + side-to-move + castling + en-passant).
- Policy model: small convolutional net (default) producing logits over a fixed UCI move vocabulary.
- Optional GPT-style tokenizer + GPT wrapper (experimental) for sequence-based modeling (see `GPTMoveTokenizer`, `GPTPolicy`).
- Baselines: Alpha-beta minimax (piece-square tables + mobility) and a lightweight MCTS (random rollouts + static eval).
- Training loop: mixed precision (AMP), gradient clipping, cosine LR with warmup, early stopping, top-1 & top-k metrics, artifact management.
- Evaluation:
  - JSONL dataset evaluation (top-1 / top-k accuracy, perplexity, optional minimax agreement) → Markdown report.
  - Arena: policy vs minimax match play with alternating colors, PGN export, Wilson 95% CI.
- FastAPI service (v2) exposing `/healthz`, `/next_move`, `/mcts_move` with lazy model loading and legal move masking.
- Tests (pytest + coverage): tokenizer, PGN parsing, minimax legality, API endpoints.
- Multi-stage Docker image (Python 3.11) with minimal runtime layer.

## Repository Structure
```
movemaster/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile                 # Multi-stage builder/runtime
├── artifacts/                 # (Optional) Saved models (e.g., policy_best.pt)
├── scripts/
│   └── preprocess.sh          # CSV -> NPZ shard (with minimax label gen)
├── src/movemaster/
│   ├── api/
│   │   ├── app.py             # FastAPI v2 (next_move, mcts_move)
│   │   └── server.py          # Legacy API (/predict, /analyze)
│   ├── service/api.py         # Service entrypoint re-export
│   ├── data/
│   │   ├── preprocess.py      # CSV preprocessing & feature encoding
│   │   └── datasets.py        # NPZ shard iterator
│   ├── models/
│   │   ├── policy.py          # Conv policy + GPT tokenizer/wrapper
│   │   └── minimax.py         # Alpha-beta with PST + choose_move
│   ├── training/train.py      # Enhanced training loop
│   ├── eval/
│   │   ├── evaluate.py        # JSONL metrics & report
│   │   └── arena.py           # Policy vs minimax arena
│   └── tests/...              # (Runtime installed via root tests/)
├── tests/                     # Pytest test suite
└── data/
    ├── raw/                   # Raw CSV (gitignored)
    └── processed/             # NPZ shards (gitignored)
```

## Data Pipeline (Kaggle Chess Evaluations)
The project assumes a CSV (e.g., `chessData.csv`) with at least:
- `FEN` — position.
- `Evaluation` — engine centipawn score (optional for training; used for future enhancements).

If a move label column (e.g., `BestMove`) is missing, we generate a next-move label via a shallow minimax search using `--minimax-depth`.

### Preprocess
```
bash scripts/preprocess.sh data/raw/chessData.csv data/processed/shard_000.npz 100000 2
```
Args: CSV path, output NPZ, max samples, minimax depth for label generation.

This writes a compressed NPZ with:
- `features`: (N, 18, 8, 8)
- `labels`: (N,) indices into the fixed UCI move vocabulary.

## Training
Enhanced training loop (conv policy):
```
python -m src.movemaster.training.train \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 256 \
  --lr 3e-4 \
  --amp \
  --grad-clip 1.0 \
  --patience 4 \
  --topk 3 \
  --warmup 2 \
  --artifacts-dir artifacts
```
Artifacts saved to:
- `artifacts/policy_best.pt` (best val top-1 accuracy)
- `artifacts/policy_last.pt` (last epoch)

## Evaluation
### JSONL Metrics
Prepare a JSONL with lines: `{ "fen": "...", "next_move": "e2e4" }`
```
python -m src.movemaster.eval.evaluate \
  --jsonl data/processed/test_positions.jsonl \
  --topk 3 \
  --report artifacts/eval_report.md
```
Outputs top-1 accuracy, top-k accuracy, perplexity, optional minimax agreement.

### Arena (Policy vs Minimax)
```
python -m src.movemaster.eval.arena \
  --games 20 \
  --minimax-depth 3 \
  --move-limit 200 \
  --out-dir artifacts/arena
```
Generates PGNs and `arena_report.md` with W/L/D and Wilson 95% CI.

## FastAPI Service
Primary app: `src.movemaster.api.app:app`

Endpoints:
- `GET /healthz` → `{ "status": "ok" }`
- `POST /next_move` → body: `{ fen: str, top_k: int }` → top-k moves with probabilities
- `POST /mcts_move` → body: `{ fen: str, n_simulations: int, max_rollout_depth?: int, c_puct?: float }`

Example (local):
```
uvicorn src.movemaster.api.app:app --host 0.0.0.0 --port 8000
```
```
curl -s -X POST localhost:8000/next_move \
  -H 'Content-Type: application/json' \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","top_k":5}'
```

Model path override: `MOVEMASTER_MODEL=artifacts/policy_best.pt` environment variable.

Legacy endpoints (`/predict`, `/analyze`) remain in `server.py` but are superseded by `/next_move` and `/mcts_move`.

## Minimax & MCTS
- Minimax uses alpha-beta with piece-square tables & mobility; select depth via `depth` argument.
- `choose_move(fen, depth=3)` returns best move UCI.
- MCTS (service only) uses random rollouts + static evaluation for approximate search.

## GPT Tokenizer (Experimental)
`GPTMoveTokenizer` & `GPTPolicy` (in `policy.py`) provide a move+FEN hashed token space. Not wired into training script yet; can be integrated for sequence-based modeling.

## Testing
```
pip install -r requirements.txt
pytest -q --cov=src/movemaster --cov-report=term-missing
```
Covers: tokenizer, PGN parsing, minimax legality, API endpoints.

## Docker
Multi-stage build:
```
docker build -t movemaster:latest .
```
Run:
```
docker run --rm -p 8000:8000 movemaster:latest
```
Override model file (mount host artifacts):
```
docker run --rm -p 8000:8000 \
  -e MOVEMASTER_MODEL=/app/artifacts/policy_best.pt \
  -v $(pwd)/artifacts:/app/artifacts \
  movemaster:latest
```

## Quick Start Summary
1. Download Kaggle CSV to `data/raw/chessData.csv`.
2. Preprocess: `bash scripts/preprocess.sh data/raw/chessData.csv data/processed/shard_000.npz 50000 2`
3. Train model (see command above) → artifacts/policy_best.pt.
4. Evaluate JSONL or run arena.
5. Serve API (local or Docker) and query `/next_move`.

## Roadmap Ideas
- Integrate GPTPolicy into training & inference.
- Add temperature / nucleus sampling for move generation.
- Quiescence search & iterative deepening for minimax.
- Stronger MCTS with neural policy priors & value head.
- ELO estimation via SPRT.

## License & Data
- Kaggle dataset usage subject to its license—verify before redistribution.
- Code provided as-is (add appropriate license file if distributing).

## Troubleshooting
| Issue | Hint |
|-------|------|
| No `policy_best.pt` found | Train first or set `MOVEMASTER_MODEL` to existing checkpoint. |
| /next_move returns empty moves | Position likely terminal; check FEN. |
| Slow minimax | Reduce depth (2) or move limit; consider pruning enhancements. |
| High Docker image size | Remove dev deps / tests or switch base to `python:3.11-slim` (already used). |

---
MoveMaster: compact chess move prediction & evaluation toolkit.
