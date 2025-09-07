from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TrainConfig:
    data_dir: str
    batch_size: int = 256
    lr: float = 3e-4
    epochs: int = 5
    device: Optional[str] = None
    ckpt_path: str = "models/policy.pt"


def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
