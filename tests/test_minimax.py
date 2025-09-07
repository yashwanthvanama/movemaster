import chess
from src.movemaster.models.minimax import choose_move


def test_minimax_returns_legal_move():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    mv = choose_move(fen, depth=2)
    assert len(mv) >= 4
    board = chess.Board(fen)
    assert chess.Move.from_uci(mv) in board.legal_moves
