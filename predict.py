"""
predict.py - Run inference on a single image or folder of images.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --folder path/to/images/
    python predict.py --demo                  # run on random CIFAR-10 test samples
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_model
from utils.data_loader import CIFAR10_MEAN, CIFAR10_STD, CLASSES


INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN Inference")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--image",  type=str, default=None, help="Path to a single image")
    parser.add_argument("--folder", type=str, default=None, help="Path to a folder of images")
    parser.add_argument("--demo",   action="store_true",   help="Demo on random CIFAR-10 samples")
    parser.add_argument("--top-k", type=int, default=3,   help="Show top-K predictions")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/predictions.png")
    return parser.parse_args()


def get_device(override=None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})
    model = build_model(
        num_classes=cfg.get("model", {}).get("num_classes", 10),
        dropout=0.0,          # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Loaded] Checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"| Val Acc: {ckpt.get('val_acc', 0):.2f}%")
    return model


def predict_tensor(model, tensor: torch.Tensor, device: torch.device, top_k: int = 3):
    """Run inference on a (1, C, H, W) tensor, return list of (class, prob)."""
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs  = torch.softmax(logits, dim=1)[0].cpu()
    top = probs.topk(top_k)
    return [(CLASSES[i], float(p)) for i, p in zip(top.indices, top.values)]


def predict_file(model, path: str, device: torch.device, top_k: int = 3):
    img = Image.open(path).convert("RGB")
    tensor = INFER_TRANSFORM(img)
    return predict_tensor(model, tensor, device, top_k), img


def demo_cifar10(model, device: torch.device, n: int = 8, output: str = "results/predictions.png"):
    """Run predictions on random CIFAR-10 test samples and save a figure."""
    import torchvision
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=INFER_TRANSFORM,
    )
    indices = torch.randperm(len(test_set))[:n]
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    mean = np.array(CIFAR10_MEAN)
    std  = np.array(CIFAR10_STD)

    fig, axes = plt.subplots(2, n // 2, figsize=(n * 1.8, 6))
    axes = axes.flatten()

    for ax_i, idx in enumerate(indices):
        img_t, true_lbl = test_set[int(idx)]
        preds = predict_tensor(model, img_t, device, top_k=1)
        pred_cls, prob  = preds[0]
        correct = (pred_cls == CLASSES[true_lbl])

        img_np = img_t.numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 1)

        axes[ax_i].imshow(img_np)
        color = "green" if correct else "red"
        axes[ax_i].set_title(
            f"GT: {CLASSES[true_lbl]}\nPred: {pred_cls} ({prob:.1%})",
            fontsize=8, color=color,
        )
        axes[ax_i].axis("off")

    plt.suptitle("CIFAR-10 Demo Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Demo predictions -> {output}")


def main():
    args   = parse_args()
    device = get_device(args.device)

    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        print("  Run `python train.py` first to train a model.")
        sys.exit(1)

    model = load_model(args.checkpoint, device)

    if args.demo:
        demo_cifar10(model, device, n=8, output=args.output)

    elif args.image:
        if not os.path.exists(args.image):
            print(f"[Error] Image not found: {args.image}")
            sys.exit(1)
        preds, _ = predict_file(model, args.image, device, args.top_k)
        print(f"\nPredictions for: {args.image}")
        for rank, (cls, prob) in enumerate(preds, 1):
            print(f"  #{rank}  {cls:<12}  {prob:.2%}")

    elif args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in os.listdir(args.folder)
                 if os.path.splitext(f)[1].lower() in exts]
        if not files:
            print(f"[Error] No image files found in {args.folder}")
            sys.exit(1)
        print(f"\nRunning inference on {len(files)} images in '{args.folder}':\n")
        for fname in sorted(files):
            path = os.path.join(args.folder, fname)
            preds, _ = predict_file(model, path, device, top_k=1)
            cls, prob = preds[0]
            print(f"  {fname:<30}  -> {cls:<12} ({prob:.2%})")

    else:
        print("No input specified. Use --image, --folder, or --demo")
        print("Example: python predict.py --demo")


if __name__ == "__main__":
    main()
