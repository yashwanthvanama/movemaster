import argparse
import os
import random
from typing import Tuple

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
        xs, ys = [], []
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
    n_val = max(1, int(n * val_frac))
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    from torch.utils.data import Subset
    return Subset(ds, train_idx), Subset(ds, val_idx)


def train_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, scaler: torch.cuda.amp.GradScaler | None, use_amp: bool, grad_clip: float) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    steps = 0
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(1, steps)


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, topk: int = 3) -> Tuple[float, float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_items = 0
    correct_top1 = 0
    correct_topk = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.detach().cpu()) * len(y)
            total_items += len(y)
            preds = logits.argmax(dim=1)
            correct_top1 += int((preds == y).sum())
            if topk > 1:
                tk_vals, tk_idx = torch.topk(logits, k=min(topk, logits.size(1)), dim=1)
                match_any = (tk_idx == y.unsqueeze(1)).any(dim=1)
                correct_topk += int(match_any.sum())
            else:
                correct_topk += int((preds == y).sum())
    avg_loss = total_loss / max(1, total_items)
    top1_acc = correct_top1 / max(1, total_items)
    topk_acc = correct_topk / max(1, total_items)
    return avg_loss, top1_acc, topk_acc


def main():
    p = argparse.ArgumentParser(description="Train MoveMaster policy model (enhanced loop)")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--artifacts-dir", type=str, default="artifacts")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm (0 disables)")
    p.add_argument("--amp", action="store_true", help="Use mixed precision (AMP)")
    p.add_argument("--topk", type=int, default=3, help="Top-K accuracy to track")
    p.add_argument("--warmup", type=int, default=0, help="Warmup epochs before cosine annealing")
    args = p.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    best_path = os.path.join(args.artifacts_dir, "policy_best.pt")
    last_path = os.path.join(args.artifacts_dir, "policy_last.pt")

    cfg = TrainConfig(data_dir=args.data_dir, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, ckpt_path=best_path)
    device = get_device(cfg)

    full = NPZDataset(cfg.data_dir)
    train_ds, val_ds = split_dataset(full, val_frac=args.val_frac)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = PolicyModel().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs - args.warmup)) if cfg.epochs > 1 else None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_metric = 0.0  # we use top-1 val accuracy for early stopping criterion
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        if args.warmup and epoch <= args.warmup:
            # Linear warmup: scale lr by epoch / warmup
            for g in optim.param_groups:
                g['lr'] = cfg.lr * epoch / max(1, args.warmup)
        train_loss = train_epoch(model, train_loader, optim, device, scaler if args.amp else None, args.amp, args.grad_clip)
        val_loss, val_top1, val_topk = eval_epoch(model, val_loader, device, topk=args.topk)
        current_lr = optim.param_groups[0]['lr']
        print(f"epoch {epoch:02d} | lr={current_lr:.5g} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_top1={val_top1:.4f} | val_top{args.topk}={val_topk:.4f}")

        improved = val_top1 > best_metric
        if improved:
            best_metric = val_top1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  saved best model -> {best_path}")
        else:
            epochs_no_improve += 1

        # Always save last state
        torch.save(model.state_dict(), last_path)

        # Step scheduler after epoch (post-warmup)
        if scheduler and (not args.warmup or epoch > args.warmup):
            scheduler.step()

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (no improvement for {args.patience} epochs).")
            break

    print(f"Best val_top1={best_metric:.4f} (saved at {best_path})")


if __name__ == "__main__":
    main()
