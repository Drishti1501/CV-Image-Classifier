# CIFAR-10 CNN Image Classifier

A convolutional neural network (CNN) for image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, built with **PyTorch**.

The model achieves **~88-90% test accuracy** using a custom CNN architecture with residual connections, data augmentation, label smoothing, and cosine-annealing LR scheduling.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Setup](#setup)
- [Usage](#usage)
  - [Train](#1-train)
  - [Evaluate](#2-evaluate)
  - [Predict](#3-predict)
- [Results](#results)
- [Configuration](#configuration)

---

## Project Structure

```
cv-image-classifier/
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── models/
│   └── cnn_model.py         # CNN architecture (ConvBlock + ResidualBlock)
├── utils/
│   ├── data_loader.py       # CIFAR-10 download, augmentation, DataLoaders
│   └── visualize.py         # Plotting: curves, confusion matrix, Grad-CAM
├── train.py                 # Training entry-point
├── evaluate.py              # Evaluation + metrics + visualizations
├── predict.py               # Inference on custom images / demo mode
├── requirements.txt
└── README.md
```

---

## Architecture Overview

```
Input (3 × 32 × 32)
   │
   ├─ Stage 1: Conv(3→64) → Conv(64→64) → ResBlock(64) → MaxPool → 16×16
   ├─ Stage 2: Conv(64→128) → Conv(128→128) → ResBlock(128) → MaxPool → 8×8
   ├─ Stage 3: Conv(128→256) → Conv(256→256) → ResBlock(256) → MaxPool → 4×4
   │
   ├─ Global Average Pooling → 256-d
   └─ FC(256→256) → ReLU → Dropout(0.4) → FC(256→10)
```

Each `ResidualBlock` adds a skip connection: `output = ReLU(conv2(conv1(x)) + x)`.
Training uses **label smoothing (0.1)**, **AdamW**, and **Cosine Annealing LR**.

---

## Setup

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Drishti1501/cv-image-classifier.git
cd cv-image-classifier
```

### 2. Create a virtual environment (recommended)

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (optional):** If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command. The code auto-detects CUDA/MPS/CPU.

---

## Usage

The CIFAR-10 dataset (~170 MB) is **downloaded automatically** on first run.

### 1. Train

```bash
python train.py
```

Optional overrides (override config.yaml on the fly):

```bash
python train.py --epochs 50 --lr 0.0005 --batch-size 64
python train.py --no-aug          # disable augmentation
python train.py --device cpu      # force CPU
```

Outputs:
- `checkpoints/best_model.pth`     — best checkpoint
- `results/training_curves.png`    — loss & accuracy curves
- `results/training_history.json`  — per-epoch metrics

### 2. Evaluate

```bash
python evaluate.py
```

Optional:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
python evaluate.py --results-dir results
```

Outputs:
- `results/evaluation_report.txt`   — per-class precision / recall / F1
- `results/metrics.json`            — summary metrics
- `results/confusion_matrix.png`    — normalised confusion matrix
- `results/sample_predictions.png`  — grid of 16 test predictions
- `results/gradcam.png`             — Grad-CAM heatmap

### 3. Predict

**Demo mode** (random CIFAR-10 test images, no extra files needed):

```bash
python predict.py --demo
```

**Single image:**

```bash
python predict.py --image path/to/your/image.jpg
```

**Folder of images:**

```bash
python predict.py --folder path/to/images/
```

---

## Results

| Metric              | Value     |
|---------------------|-----------|
| Test Accuracy       | ~88–90%   |
| Parameters          | ~1.4 M    |
| Training Time (GPU) | ~15 min   |
| Training Time (CPU) | ~2–3 hrs  |

Generated visualizations are saved in `results/`:

| File | Description |
|------|-------------|
| `training_curves.png`   | Loss & accuracy over epochs |
| `confusion_matrix.png`  | Normalised 10×10 confusion matrix |
| `sample_predictions.png`| 16 test samples with GT vs predicted labels |
| `gradcam.png`           | Grad-CAM saliency map |

---

## Configuration

Edit `configs/config.yaml` to change hyperparameters:

```yaml
training:
  epochs: 30
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0005
  optimizer: adam          # adam | sgd
  scheduler: cosine        # cosine | step | none
  early_stopping_patience: 10
```

---

## CIFAR-10 Classes

| ID | Class      |
|----|------------|
| 0  | airplane   |
| 1  | automobile |
| 2  | bird       |
| 3  | cat        |
| 4  | deer       |
| 5  | dog        |
| 6  | frog       |
| 7  | horse      |
| 8  | ship       |
| 9  | truck      |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
