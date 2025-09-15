#!/usr/bin/env python3
import csv
import json
import chess
from src.movemaster.models.minimax import choose_move

def generate_test_data(csv_path, output_path, num_samples=10, depth=1):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        samples = []
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            fen = row['FEN']
            try:
                board = chess.Board(fen)
                next_move = choose_move(fen, depth=depth)
                samples.append({"fen": fen, "next_move": next_move})
            except Exception as e:
                print(f"Error with FEN {fen}: {e}")
                continue
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Generated {len(samples)} test samples in {output_path}")

if __name__ == "__main__":
    generate_test_data('data/raw/chessData.csv', 'test.jsonl', num_samples=100, depth=1)
