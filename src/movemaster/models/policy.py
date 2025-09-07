from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn

from ..data.preprocess import VOCAB_SIZE, MOVE_VOCAB, moves_to_mask


class PolicyModel(nn.Module):
    def __init__(self, in_planes: int = 18, hidden: int = 128, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden), nn.ReLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(device: Optional[torch.device] = None, ckpt_path: Optional[str] = None) -> PolicyModel:
    model = PolicyModel()
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

# ------------------------------
# GPT-2 wrapper with custom tokenizer
# ------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoConfig
except ImportError:  # transformers not installed yet
    AutoModelForCausalLM = None  # type: ignore
    AutoConfig = None  # type: ignore

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
FEN_BUCKETS = 1024  # number of hashed FEN tokens
FEN_TOKENS = [f"<FEN{i}>" for i in range(FEN_BUCKETS)]
# Move tokens come directly from MOVE_VOCAB

class GPTMoveTokenizer:
    """Minimal tokenizer mapping:
    <PAD>, <BOS>, <EOS>, <UNK>, <FEN0..N>, <MOVE_UCI...>
    Each move (uci) is a single token; FEN is hashed to one of FEN_BUCKETS tokens.
    """
    def __init__(self):
        self.special = SPECIAL_TOKENS
        self.fen_tokens = FEN_TOKENS
        self.move_tokens = MOVE_VOCAB
        self.idx_to_token: List[str] = self.special + self.fen_tokens + self.move_tokens
        self.token_to_idx: Dict[str, int] = {t: i for i, t in enumerate(self.idx_to_token)}
        self.pad_id = self.token_to_idx["<PAD>"]
        self.bos_id = self.token_to_idx["<BOS>"]
        self.eos_id = self.token_to_idx["<EOS>"]
        self.unk_id = self.token_to_idx["<UNK>"]
        self.move_offset = len(self.special) + len(self.fen_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_token)

    def hash_fen(self, fen: str) -> int:
        return (hash(fen) % FEN_BUCKETS)

    def encode_fen_context(self, fen: str) -> List[int]:
        # context = BOS + hashed FEN token
        bucket = self.hash_fen(fen)
        return [self.bos_id, self.token_to_idx[FEN_TOKENS[bucket]]]

    def move_token_id(self, uci: str) -> Optional[int]:
        # Map move uci to token id
        try:
            rel = MOVE_VOCAB.index(uci)
        except ValueError:
            return None
        return self.move_offset + rel

    def decode_move(self, token_id: int) -> Optional[str]:
        if token_id >= self.move_offset and token_id < self.move_offset + len(MOVE_VOCAB):
            return MOVE_VOCAB[token_id - self.move_offset]
        return None

class GPTPolicy(nn.Module):
    """GPT-2 small wrapper for next-move prediction.

    forward(input_ids, attention_mask, labels):
        - input_ids: (B, L) context tokens (no target move token)
        - labels: (B,) target move token ids (move only)
        Returns dict(loss, logits) where loss is CE on next-move token.
    """
    def __init__(self, tokenizer: Optional[GPTMoveTokenizer] = None, model_name: str = "gpt2"):
        super().__init__()
        if AutoModelForCausalLM is None:
            raise ImportError("transformers package not installed. Add transformers to requirements.")
        self.tokenizer = tokenizer or GPTMoveTokenizer()
        # Try to load pretrained; fall back to config-only if offline.
        try:
            self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception:
            if AutoConfig is not None:
                try:
                    cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    # Define a GPT-2 small compatible config
                    cfg = AutoConfig(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12)
            else:
                raise
            self.lm = AutoModelForCausalLM(cfg)
        # Resize embeddings to custom vocab
        self.lm.resize_token_embeddings(self.tokenizer.vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        # logits: (B, L, V); we use last position (L-1) for next move prediction
        logits = out.logits  # type: ignore
        last_logits = logits[:, -1, :]  # (B, V)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(last_logits, labels)
        return {"loss": loss, "logits": last_logits}

    @torch.no_grad()
    def predict_topk(self, fen: str, k: int = 5) -> List[Tuple[str, float]]:
        import chess
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return []
        ctx_ids = self.tokenizer.encode_fen_context(fen)
        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_ids)
        out = self.lm(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits[:, -1, :][0]  # (V,)
        # Mask: only allow move tokens that correspond to legal moves
        mask = torch.full_like(logits, float('-inf'))
        move_probs: List[Tuple[str, float]] = []
        allowed_ids = []
        id_to_move: Dict[int, str] = {}
        for mv in legal_moves:
            tid = self.tokenizer.move_token_id(mv)
            if tid is not None:
                allowed_ids.append(tid)
                id_to_move[tid] = mv
        if not allowed_ids:
            return []
        mask[allowed_ids] = logits[allowed_ids]
        probs = torch.softmax(mask, dim=-1)
        # Extract top-k among allowed ids
        allowed_probs = probs[allowed_ids]
        topk = min(k, len(allowed_ids))
        top_vals, top_idx = torch.topk(allowed_probs, topk)
        for val, idx_local in zip(top_vals.tolist(), top_idx.tolist()):
            global_id = allowed_ids[idx_local]
            move_probs.append((id_to_move[global_id], float(val)))
        return move_probs


def load_gpt_policy(device: Optional[torch.device] = None) -> GPTPolicy:
    tok = GPTMoveTokenizer()
    model = GPTPolicy(tok)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model
