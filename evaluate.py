"""
evaluate.py - Evaluate a trained CNN checkpoint on CIFAR-10 test set.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import os

import torch
import numpy as np
from sklearn.metrics import classification_report

from models import build_model
from utils import get_dataloaders, CLASSES
from utils import plot_confusion_matrix, plot_sample_predictions, plot_gradcam


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 CNN")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def get_device(override=None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  CIFAR-10 CNN Classifier — Evaluation")
    print(f"{'='*55}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}\n")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", {})
    num_classes = cfg.get("model", {}).get("num_classes", 10)
    dropout     = cfg.get("model", {}).get("dropout", 0.4)

    model = build_model(num_classes=num_classes, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Data
    data_dir    = cfg.get("data", {}).get("data_dir", "./data")
    num_workers = cfg.get("data", {}).get("num_workers", 2)
    _, test_loader = get_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size,
        num_workers=num_workers, augment=False,
    )

    # Inference
    all_preds, all_labels, all_images = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_images.append(images.cpu())

    all_images = torch.cat(all_images)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / len(all_labels)

    print(f"  Test Accuracy : {accuracy:.2f}%  ({correct}/{len(all_labels)})\n")
    print("  Per-class Report:")
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
    print(report)

    # Save report
    report_path = os.path.join(args.results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    print(f"  [Saved] Report -> {report_path}")

    # Metrics JSON
    metrics = {"test_accuracy": accuracy, "num_correct": correct, "total": len(all_labels)}
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Visualizations
    plot_confusion_matrix(
        all_labels, all_preds,
        save_path=os.path.join(args.results_dir, "confusion_matrix.png"),
    )
    plot_sample_predictions(
        all_images[:16], all_labels[:16], all_preds[:16],
        save_path=os.path.join(args.results_dir, "sample_predictions.png"),
    )

    # Grad-CAM on first test image
    try:
        plot_gradcam(
            model, all_images[0].to(device),
            true_label=all_labels[0], pred_label=all_preds[0],
            save_path=os.path.join(args.results_dir, "gradcam.png"),
        )
    except Exception as exc:
        print(f"  [Warning] Grad-CAM skipped: {exc}")

    print(f"\n  All results saved to ./{args.results_dir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
