import pytest
from src.movemaster.models.policy import GPTMoveTokenizer


def test_tokenizer_basic_encode_decode():
    tok = GPTMoveTokenizer()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ctx_ids = tok.encode_fen_context(fen)
    assert len(ctx_ids) == 2  # BOS + hashed FEN bucket
    # Ensure vocab size covers moves
    assert tok.vocab_size > 4000


def test_tokenizer_move_id_round_trip():
    tok = GPTMoveTokenizer()
    sample_move = "e2e4"
    tid = tok.move_token_id(sample_move)
    assert tid is not None
    decoded = tok.decode_move(tid)
    assert decoded == sample_move
