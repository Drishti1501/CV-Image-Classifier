"""
train.py - Train the CNN on CIFAR-10.

Usage:
    python train.py
    python train.py --config configs/config.yaml
    python train.py --epochs 50 --lr 0.001 --batch-size 64
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from models import build_model
from utils import get_dataloaders, plot_training_curves


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--no-aug",     action="store_true")
    parser.add_argument("--device",     default=None, help="cpu | cuda | mps")
    return parser.parse_args()


def get_device(override=None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.epochs:     cfg["training"]["epochs"] = args.epochs
    if args.lr:         cfg["training"]["learning_rate"] = args.lr
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.no_aug:     cfg["data"]["augmentation"] = False

    set_seed(cfg.get("seed", 42))
    device = get_device(args.device)
    print(f"\n{'='*55}")
    print(f"  CIFAR-10 CNN Classifier â€” Training")
    print(f"{'='*55}")
    print(f"  Device : {device}")
    print(f"  Epochs : {cfg['training']['epochs']}")
    print(f"  LR     : {cfg['training']['learning_rate']}")
    print(f"  Batch  : {cfg['training']['batch_size']}")
    print(f"{'='*55}\n")

    # Paths
    ckpt_dir    = cfg["paths"]["checkpoints"]
    results_dir = cfg["paths"]["results"]
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Data
    train_loader, test_loader = get_dataloaders(
        data_dir    = cfg["data"]["data_dir"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        augment     = cfg["data"]["augmentation"],
    )

    # Model
    model = build_model(
        num_classes = cfg["model"]["num_classes"],
        dropout     = cfg["model"]["dropout"],
    ).to(device)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    opt_name = cfg["training"]["optimizer"].lower()
    if opt_name == "adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            momentum=cfg["training"].get("momentum", 0.9),
            weight_decay=cfg["training"]["weight_decay"],
            nesterov=True,
        )

    sched_name = cfg["training"]["scheduler"].lower()
    if sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    elif sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # Training loop
    epochs    = cfg["training"]["epochs"]
    patience  = cfg["training"]["early_stopping_patience"]
    best_acc  = 0.0
    no_improve = 0

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, test_loader, criterion, device)
        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{epochs}] "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
        })

        # Checkpoint
        if va_acc > best_acc:
            best_acc = va_acc
            no_improve = 0
            ckpt = {
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_acc, "config": cfg,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  -> New best model saved! Val Acc: {best_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Early stopping] No improvement for {patience} epochs.")
                break

    # Save training history
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(results_dir, "training_curves.png"),
    )

    print(f"\n{'='*55}")
    print(f"  Training complete. Best Val Acc: {best_acc:.2f}%")
    print(f"  Checkpoint : {ckpt_dir}/best_model.pth")
    print(f"  Results    : {results_dir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required on Windows with num_workers > 0
    main()

