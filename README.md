# MoveMaster

MoveMaster is an AI-powered chess move prediction engine that predicts the next best move from a given chess position (FEN string). It uses a lightweight convolutional neural network trained on chess evaluation data, with baselines including minimax and Monte Carlo Tree Search (MCTS). The project includes data preprocessing, training scripts, evaluation tools, and a FastAPI web service for easy integration.

## Features
- **Data Pipeline**: Processes Kaggle Chess Evaluations CSV (FEN + evaluation scores) into training-ready NPZ shards, with optional move label generation via minimax.
- **Models**: Convolutional policy network for move prediction, optional GPT-style tokenizer and wrapper for experimental sequence modeling.
- **Baselines**: Alpha-beta minimax with piece-square tables, and lightweight MCTS for comparison.
- **Training**: Advanced loop with mixed precision, gradient clipping, cosine learning rate, early stopping, and metrics tracking.
- **Evaluation**: JSONL-based metrics (accuracy, perplexity) and arena mode for policy vs minimax matches with PGN export.
- **API**: FastAPI service with endpoints for move prediction and health checks.
- **Testing**: Comprehensive pytest suite.
- **Docker**: Multi-stage container for easy deployment.

## How to Run the Project

### Prerequisites
- Python 3.10+
- Git
- (Optional) Docker

### 1. Clone and Setup Environment
```bash
git clone <repository-url>
cd movemaster
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
Download the Kaggle Chess Evaluations dataset (CSV with `FEN` and `Evaluation` columns) to `data/raw/chessData.csv`.

Preprocess the data:
```bash
bash scripts/preprocess.sh data/raw/chessData.csv data/processed/shard_000.npz 50000 2
```
This generates NPZ shards with features and labels.

### 3. Train the Model
```bash
python -m src.movemaster.training.train \
  --data-dir data/processed \
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4 \
  --amp \
  --artifacts-dir artifacts
```
Trained model saved to `artifacts/policy_best.pt`.

### 4. Evaluate (Optional)
Generate test data from your CSV (recommended for in-distribution evaluation):
```bash
python generate_test.py  # Generates 100 samples in test.jsonl
```
Or create manually:
```json
{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "next_move": "e2e4"}
```
Evaluate:
```bash
python -m src.movemaster.eval.evaluate \
  --jsonl test.jsonl \
  --topk 3 \
  --report artifacts/eval_report.md
```

### 5. Run the API Service
Locally:
```bash
uvicorn src.movemaster.api.app:app --host 0.0.0.0 --port 8000
```
Test:
```bash
curl -X POST http://localhost:8000/next_move \
  -H "Content-Type: application/json" \
  -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","top_k":5}'
```

Or via Docker:
```bash
docker build -t movemaster .
docker run -p 8000:8000 movemaster
```

### 6. Run Tests
```bash
pytest
```

## Repository Structure
- `src/movemaster/`: Core modules (API, data, models, training, eval)
- `tests/`: Test suite
- `scripts/`: Preprocessing scripts
- `data/`: Raw and processed data
- `artifacts/`: Trained models and reports
- `requirements.txt`: Dependencies
- `Dockerfile`: Container build

## License
See LICENSE file. Data from Kaggle subject to its terms.
