from typing import Optional, Tuple
import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate(board: chess.Board) -> int:
    score = 0
    for piece_type, val in PIECE_VALUES.items():
        score += val * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    # Mobility
    score += 5 * (len(list(board.legal_moves))) * (1 if board.turn == chess.WHITE else -1)
    if board.is_checkmate():
        return 10_000 if board.turn == chess.BLACK else -10_000
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0
    return score


def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool) -> Tuple[int, Optional[chess.Move]]:
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_move = None
    if maximizing:
        value = -10_000_000
        for move in board.legal_moves:
            board.push(move)
            score, _ = alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()
            if score > value:
                value = score
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = 10_000_000
        for move in board.legal_moves:
            board.push(move)
            score, _ = alphabeta(board, depth - 1, alpha, beta, True)
            board.pop()
            if score < value:
                value = score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move


def minimax_best_move(board: chess.Board, depth: int = 3) -> Tuple[Optional[chess.Move], int]:
    maximizing = board.turn == chess.WHITE
    score, move = alphabeta(board, depth, -10_000_000, 10_000_000, maximizing)
    return move, score
