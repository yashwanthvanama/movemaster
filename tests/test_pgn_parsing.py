import io
import chess.pgn
from src.movemaster.data.preprocess import board_to_features, _build_move_vocab

SAMPLE_PGN = """[Event "?"]
[Site "?"]
[Date "2024.01.01"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""

def test_pgn_parsing_and_features():
    game = chess.pgn.read_game(io.StringIO(SAMPLE_PGN))
    board = game.board()
    first_move = next(game.mainline_moves())
    board.push(first_move)
    feats = board_to_features(board)
    assert feats.shape[0] == 18 and feats.shape[1] == 8 and feats.shape[2] == 8
