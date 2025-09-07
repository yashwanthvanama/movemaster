from typing import Optional, Tuple, List
import chess

# Material values (centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King intrinsic value handled via mate scores
}

# Piece-Square Tables (mid-game approximation, centipawns)
# Indexed from White perspective (rank 8 -> 1 converted when accessing for black)
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_PST = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_PST = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK_PST = [
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
QUEEN_PST = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
KING_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PSTS = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}

MATE_SCORE = 100000


def _pst_score(piece_type: chess.PieceType, square: int, color: chess.Color) -> int:
    table = PSTS[piece_type]
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    # Convert to index from White POV (rank 7 -> index 0 ... rank 0 -> last row)
    if color == chess.WHITE:
        r = 7 - rank
    else:
        r = rank  # mirror for black naturally by not flipping
    idx = r * 8 + file
    val = table[idx]
    return val if color == chess.WHITE else -val


def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0

    score = 0
    # Material + PST
    for piece_type, base in PIECE_VALUES.items():
        for sq in board.pieces(piece_type, chess.WHITE):
            score += base + _pst_score(piece_type, sq, chess.WHITE)
        for sq in board.pieces(piece_type, chess.BLACK):
            score -= base + _pst_score(piece_type, sq, chess.BLACK)

    # Mobility (small weight)
    mobility = len(list(board.legal_moves))
    score += 5 * (mobility if board.turn == chess.WHITE else -mobility)
    return score


def _move_order(board: chess.Board):
    moves = list(board.legal_moves)
    # Simple ordering: captures first (MVV-LVA-ish by captured piece value), then others
    scored = []
    for m in moves:
        if board.is_capture(m):
            captured = board.piece_at(m.to_square)
            val = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
            scored.append((1000 + val, m))
        else:
            scored.append((0, m))
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[int, Optional[chess.Move]]:
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_move = None
    maximizing = board.turn == chess.WHITE
    value = -MATE_SCORE if maximizing else MATE_SCORE

    for move in _move_order(board):
        board.push(move)
        score, _ = alphabeta(board, depth - 1, alpha, beta)
        board.pop()
        if maximizing:
            if score > value:
                value = score
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        else:
            if score < value:
                value = score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
    return value, best_move


def minimax_best_move(board: chess.Board, depth: int = 3) -> Tuple[Optional[chess.Move], int]:
    score, move = alphabeta(board, depth, -MATE_SCORE, MATE_SCORE)
    return move, score


def choose_move(fen: str, depth: int = 3) -> str:
    """Return best move in UCI for given FEN (empty string if none)."""
    board = chess.Board(fen)
    move, _ = minimax_best_move(board, depth=depth)
    return move.uci() if move else ""
