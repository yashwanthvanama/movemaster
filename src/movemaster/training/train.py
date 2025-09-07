import argparse
import os
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..data.datasets import iter_npz_shards
from ..data.preprocess import VOCAB_SIZE
from ..models.policy import PolicyModel
from .utils import TrainConfig, get_device


class NPZDataset(Dataset):
    def __init__(self, data_dir: str):
        xs = []
        ys = []
        for X, y in iter_npz_shards(data_dir):
            xs.append(X)
            ys.append(y)
        if not xs:
            raise FileNotFoundError(f"No npz shards found under {data_dir}")
        self.X = np.concatenate(xs, 0)
        self.y = np.concatenate(ys, 0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = int(self.y[idx])
        return x, y


def split_dataset(ds: Dataset, val_frac: float = 0.05):
    n = len(ds)
    idxs = list(range(n))
    random.shuffle(idxs)
    n_val = int(n * val_frac)
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    from torch.utils.data import Subset
    return Subset(ds, train_idx), Subset(ds, val_idx)


def train_epoch(model, loader, optim, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(1, count)


def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += len(y)
    return correct / max(1, total)


def main():
    p = argparse.ArgumentParser(description="Train MoveMaster policy model")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=str, default="models/policy.pt")
    args = p.parse_args()

    cfg = TrainConfig(data_dir=args.data_dir, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, ckpt_path=args.out)
    device = get_device(cfg)

    full = NPZDataset(cfg.data_dir)
    train_ds, val_ds = split_dataset(full, val_frac=0.05)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = PolicyModel().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    for epoch in range(cfg.epochs):
        loss = train_epoch(model, train_loader, optim, device)
        acc = eval_epoch(model, val_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg.ckpt_path)
            print(f"saved best to {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
