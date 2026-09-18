"""
Visualization utilities:
  - Training curves (loss & accuracy)
  - Confusion matrix
  - Sample predictions grid
  - Grad-CAM heatmap
"""

import os
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for CLI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────── Training Curves ────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str,
):
    """Save loss and accuracy curves to *save_path*."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, val_losses,   "r-o", markersize=4, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", markersize=4, label="Train Acc")
    ax2.plot(epochs, val_accs,   "r-o", markersize=4, label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curve"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Training curves -> {save_path}")


# ─────────────────────────── Confusion Matrix ───────────────────────────────

def plot_confusion_matrix(
    true_labels: List[int],
    pred_labels: List[int],
    save_path: str,
    class_names: Optional[List[str]] = None,
):
    """Save normalised confusion matrix to *save_path*."""
    if class_names is None:
        class_names = CLASSES
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Confusion matrix -> {save_path}")


# ─────────────────────────── Sample Predictions ─────────────────────────────

def plot_sample_predictions(
    images: torch.Tensor,
    true_labels: List[int],
    pred_labels: List[int],
    save_path: str,
    class_names: Optional[List[str]] = None,
    n: int = 16,
):
    """Save a grid of up to *n* sample predictions with colour-coded titles."""
    if class_names is None:
        class_names = CLASSES

    n = min(n, len(images))
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    axes = axes.flatten()

    # Denormalise
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * std + mean).clip(0, 1)
        correct = (true_labels[i] == pred_labels[i])
        color = "green" if correct else "red"
        axes[i].imshow(img)
        axes[i].set_title(
            f"T:{class_names[true_labels[i]]}\nP:{class_names[pred_labels[i]]}",
            fontsize=7, color=color,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    correct_patch = mpatches.Patch(color="green", label="Correct")
    wrong_patch   = mpatches.Patch(color="red",   label="Wrong")
    fig.legend(handles=[correct_patch, wrong_patch], loc="lower right", fontsize=10)
    plt.suptitle("Sample Predictions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Sample predictions -> {save_path}")


# ─────────────────────────── Grad-CAM ───────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Usage:
        cam = GradCAM(model, target_layer=model.stage3[-3])
        heatmap = cam(input_tensor, class_idx)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def plot_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    true_label: int,
    pred_label: int,
    save_path: str,
    class_names: Optional[List[str]] = None,
):
    """Overlay Grad-CAM heatmap on a single CIFAR-10 image and save."""
    if class_names is None:
        class_names = CLASSES

    # Identify last Conv layer (stage3's second ConvBlock's conv layer)
    target_layer = model.stage3[1].block[0]   # ConvBlock -> nn.Sequential -> Conv2d
    cam_extractor = GradCAM(model, target_layer)
    heatmap = cam_extractor(image.unsqueeze(0), pred_label)

    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * std + mean).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(img_np);        axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    overlay = img_np * 0.6 + plt.cm.jet(heatmap)[..., :3] * 0.4
    axes[2].imshow(overlay.clip(0, 1)); axes[2].axis("off")
    axes[2].set_title(f"Overlay\nTrue:{class_names[true_label]} | Pred:{class_names[pred_label]}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Grad-CAM -> {save_path}")
