from typing import Optional
import torch
import torch.nn as nn

from ..data.preprocess import VOCAB_SIZE


class PolicyModel(nn.Module):
    def __init__(self, in_planes: int = 18, hidden: int = 128, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        # Simple ConvNet -> MLP head
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
