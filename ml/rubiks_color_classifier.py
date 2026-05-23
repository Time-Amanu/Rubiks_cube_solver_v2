"""
rubiks_color_classifier.py
===========================
Fine-tunes MobileNetV3-Small (or ResNet-18) on a Kaggle dataset of
Rubik's-cube sticker images to classify 6 sticker colors.

COLOR ENCODING  (matches the FastAPI solver contract)
──────────────────────────────────────────────────────
    0 = White     1 = Yellow    2 = Green
    3 = Blue      4 = Red       5 = Orange

EXPECTED DATASET LAYOUT
────────────────────────
Any ImageFolder-compatible layout works:

    data/
    ├── train/
    │   ├── white/    (or 0/)
    │   ├── yellow/
    │   ├── green/
    │   ├── blue/
    │   ├── red/
    │   └── orange/
    └── val/
        ├── white/
        └── ...

The folder names are case-insensitive; the script maps them to 0-5 via
CLASS_TO_IDX (see below).  If your Kaggle dataset uses numeric folders
(0-5), set USE_NUMERIC_FOLDERS = True.

QUICK START
───────────
    pip install torch torchvision matplotlib pillow tqdm

    # Train
    python rubiks_color_classifier.py --mode train \
        --data_dir ./data --epochs 15 --model mobilenet

    # Inference on a single 2×2 face photo
    python rubiks_color_classifier.py --mode infer \
        --image ./face.jpg --checkpoint ./best_model.pt

    # Debug misclassifications on the val set
    python rubiks_color_classifier.py --mode debug \
        --data_dir ./data --checkpoint ./best_model.pt
"""

from __future__ import annotations

import argparse
import time
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to "TkAgg" if you have a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder

from PIL import Image, ImageDraw

# ─── Configuration ────────────────────────────────────────────────────────────

# Map class name → integer color index.
# Add aliases for however your Kaggle dataset labels its folders.
CLASS_TO_IDX: dict[str, int] = {
    # Text names
    "white":  0, "yellow": 1, "green":  2,
    "blue":   3, "red":    4, "orange": 5,
    # Numeric names (if dataset uses 0-5 folder names)
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    # Common Kaggle variants
    "w": 0, "y": 1, "g": 2, "b": 3, "r": 4, "o": 5,
}

IDX_TO_CLASS: dict[int, str] = {
    0: "White", 1: "Yellow", 2: "Green",
    3: "Blue",  4: "Red",    5: "Orange",
}

# Sticker color for matplotlib visualisation
IDX_TO_HEX: dict[int, str] = {
    0: "#FFFFFF",   # White
    1: "#FFD700",   # Yellow
    2: "#00AA44",   # Green
    3: "#0055CC",   # Blue
    4: "#CC2200",   # Red
    5: "#FF6600",   # Orange
}

NUM_CLASSES = 6
IMG_SIZE    = 224       # standard ImageNet input size

# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(mode: str = "train") -> T.Compose:
    """
    Return augmentation pipeline for training or evaluation.

    Training augmentations are chosen for sticker images:
      • Color jitter captures lighting variations on plastic faces.
      • RandomPerspective handles slight camera angles.
      • Moderate rotation handles hand-held capture tilt.
      • We do NOT flip horizontally — color patches are symmetric,
        so flipping won't hurt, but we keep it for safety.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),   # slightly larger for crop
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(
                brightness=0.4,   # stickers vary a lot under different light
                contrast=0.3,
                saturation=0.4,
                hue=0.05,         # small hue shift — colors must stay recognisable
            ),
            T.RandomPerspective(distortion_scale=0.2, p=0.4),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:  # val / test / inference
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ─── Dataset ─────────────────────────────────────────────────────────────────

class RubiksDataset(ImageFolder):
    """
    Thin wrapper around torchvision's ImageFolder that:
      1. Remaps folder names to our canonical 0-5 integer labels via CLASS_TO_IDX.
      2. Exposes the original file path for every sample (needed by the
         misclassification debugger).
      3. Keeps a ``class_counts`` attribute for computing sampler weights.

    Args:
        root: Path to ``train/`` or ``val/`` directory.
        transform: Torchvision transform pipeline.
        target_transform: Optional label transform (unused by default).
    """

    def __init__(self, root: str | Path, transform=None, target_transform=None):
        super().__init__(str(root), transform=transform,
                         target_transform=target_transform)
        # Remap ImageFolder's auto-assigned class indices to our canonical ones
        self._remap_targets()
        self.class_counts = self._compute_class_counts()

    # ── Internals ──────────────────────────────────────────────────────────

    def _remap_targets(self) -> None:
        """
        ImageFolder assigns class indices alphabetically (blue=0, green=1 …).
        We overwrite those with CLASS_TO_IDX so the model always outputs
        White=0, Yellow=1, Green=2, Blue=3, Red=4, Orange=5 — regardless of
        how the filesystem sorts the folders.
        """
        new_targets = []
        for (path, orig_idx) in self.samples:
            folder_name = Path(path).parent.name.lower()
            if folder_name in CLASS_TO_IDX:
                canonical = CLASS_TO_IDX[folder_name]
            else:
                # Fall back to the original ImageFolder index with a warning
                print(f"[WARN] Unknown class folder '{folder_name}'; keeping idx {orig_idx}")
                canonical = orig_idx
            new_targets.append(canonical)

        # Rebuild samples list with remapped labels
        self.samples  = [(p, t) for (p, _), t in zip(self.samples, new_targets)]
        self.targets  = new_targets
        self.imgs     = self.samples  # ImageFolder alias

    def _compute_class_counts(self) -> list[int]:
        counts = [0] * NUM_CLASSES
        for _, label in self.samples:
            counts[label] += 1
        return counts

    # ── Public helpers ─────────────────────────────────────────────────────

    def get_sampler_weights(self) -> list[float]:
        """
        Per-sample weights for WeightedRandomSampler.
        Inverts class frequency so rare colors are oversampled.
        """
        class_weight = [
            1.0 / max(c, 1) for c in self.class_counts
        ]
        return [class_weight[label] for _, label in self.samples]

    def __getitem__(self, index: int):
        """Returns (image_tensor, label, file_path) — path needed for debugging."""
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    Args:
        data_dir: Root directory containing ``train/`` and ``val/`` subdirectories.
        batch_size: Samples per mini-batch.
        num_workers: Parallel data-loading workers (set 0 on Windows).
        use_weighted_sampler: Oversample rare classes in the training set.

    Returns:
        (train_loader, val_loader)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Expected a 'train' subdirectory inside '{data_dir}'. "
            f"Found: {list(data_dir.iterdir())}"
        )

    train_dataset = RubiksDataset(train_dir, transform=get_transforms("train"))
    val_dataset   = RubiksDataset(val_dir,   transform=get_transforms("val"))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val   samples: {len(val_dataset)}")
    print(f"Class counts  (train): { {IDX_TO_CLASS[i]: n for i, n in enumerate(train_dataset.class_counts)} }")

    # Weighted sampler balances under-represented colors during training
    if use_weighted_sampler:
        weights = train_dataset.get_sampler_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,          # avoid partial batches in BatchNorm layers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# ─── Model ───────────────────────────────────────────────────────────────────

def build_model(
    arch: str = "mobilenet",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Load a pre-trained backbone and replace its classifier head.

    Supported architectures
    ───────────────────────
    mobilenet  → MobileNetV3-Small  (~2.5 M params, fastest)
    resnet18   → ResNet-18          (~11 M params, more accurate)
    resnet50   → ResNet-50          (~25 M params, best accuracy)

    The backbone's first layers are frozen for the first few epochs
    (controlled by the ``freeze_backbone`` flag in the training loop),
    then unfrozen for fine-tuning at a lower learning rate.
    """
    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if arch == "mobilenet":
        model = models.mobilenet_v3_small(weights=weights_arg)
        # MobileNetV3-Small classifier: [Linear(576→1024), Hardswish, Dropout, Linear(1024→1000)]
        in_features = model.classifier[0].in_features   # 576
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    elif arch == "resnet18":
        model = models.resnet18(weights=weights_arg)
        in_features = model.fc.in_features              # 512
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "resnet50":
        model = models.resnet50(weights=weights_arg)
        in_features = model.fc.in_features              # 2048
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose: mobilenet, resnet18, resnet50")

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {arch}  |  Total params: {total_params:,}  |  Trainable: {trainable_p:,}")
    return model


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Freeze all layers except the final classifier head."""
    if arch == "mobilenet":
        head_attr = "classifier"
    else:                           # resnet*
        head_attr = "fc"

    for name, param in model.named_parameters():
        if not name.startswith(head_attr):
            param.requires_grad = False

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  ↳ Backbone frozen ({frozen:,} params) — head-only phase")


def unfreeze_all(model: nn.Module) -> None:
    """Re-enable gradients for all parameters (full fine-tuning)."""
    for param in model.parameters():
        param.requires_grad = True
    print("  ↳ Full model unfrozen — fine-tuning phase")


# ─── Training loop ────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running mean of a scalar (loss, accuracy, …)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy as a Python float in [0, 1]."""
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    phase: str = "train",
) -> tuple[float, float]:
    """
    Run one epoch of training or validation.

    Args:
        model:     The network.
        loader:    DataLoader yielding (image, label, path) triples.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Adam optimizer (None during validation).
        device:    CUDA or CPU.
        phase:     "train" or "val".

    Returns:
        (mean_loss, mean_accuracy)
    """
    is_train = phase == "train"
    model.train(is_train)

    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images, labels, _paths = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)                      # (B, 6)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                # Gradient clipping prevents exploding gradients on small datasets
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = images.size(0)
            loss_meter.update(loss.item(), bs)
            acc_meter.update(accuracy(outputs.detach(), labels), bs)

    return loss_meter.avg, acc_meter.avg


def train(
    data_dir: str | Path,
    checkpoint_path: str | Path = "best_model.pt",
    arch: str = "mobilenet",
    epochs: int = 20,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    head_only_epochs: int = 3,      # freeze backbone for first N epochs
    num_workers: int = 4,
    seed: int = 42,
    dropout: float = 0.3,
) -> dict:
    """
    Full training procedure.

    Strategy
    ────────
    1. Head-only phase (``head_only_epochs``):
       Freeze the ImageNet backbone; train only the new classifier head at
       ``lr_head`` until it converges enough that the gradient signal is clean.
    2. Fine-tuning phase:
       Unfreeze the whole network; train at the lower ``lr_backbone``.
       CosineAnnealingLR smoothly decays the learning rate to near-zero.

    This two-phase approach gives better accuracy than fine-tuning from scratch
    and avoids destroying the pre-trained representations early on.

    Args:
        data_dir:         Root dir with train/ and val/ subdirs.
        checkpoint_path:  Where to save the best model weights.
        arch:             "mobilenet" | "resnet18" | "resnet50".
        epochs:           Total training epochs (including head-only phase).
        batch_size:       Mini-batch size.
        lr_head:          Learning rate for head-only phase.
        lr_backbone:      Learning rate for fine-tuning phase.
        head_only_epochs: Number of head-only epochs before unfreezing.
        num_workers:      DataLoader workers.
        seed:             Random seed for reproducibility.
        dropout:          Dropout rate in the classifier head.

    Returns:
        History dict: {"train_loss", "val_loss", "train_acc", "val_acc"}
    """
    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(arch=arch, dropout=dropout).to(device)

    # ── Loss & optimizer ───────────────────────────────────────────────────
    # label_smoothing=0.1 prevents overconfidence on small datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Head-only phase ────────────────────────────────────────────────────
    freeze_backbone(model, arch)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── History ────────────────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'─'*65}")
    print(f"{'Epoch':>5}  {'Phase':>10}  {'Loss':>8}  {'Acc':>8}  {'Time':>7}")
    print(f"{'─'*65}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Switch to full fine-tuning after head-only phase
        if epoch == head_only_epochs + 1:
            unfreeze_all(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr_backbone,
                weight_decay=1e-4,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - head_only_epochs)

        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, phase="train"
        )
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, phase="val"
        )

        elapsed = time.time() - t0
        phase_label = "head" if epoch <= head_only_epochs else "finetune"

        print(
            f"{epoch:>5}  {phase_label:>10}  "
            f"{train_loss:>8.4f}  {train_acc*100:>7.2f}%  {elapsed:>5.1f}s"
            f"  │  val {val_loss:.4f}  {val_acc*100:.2f}%"
            + ("  ★" if val_acc > best_val_acc else "")
        )

        # ── Record ─────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":      epoch,
                    "arch":       arch,
                    "state_dict": model.state_dict(),
                    "val_acc":    val_acc,
                    "class_to_idx": CLASS_TO_IDX,
                },
                checkpoint_path,
            )

    print(f"{'─'*65}")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%  →  saved to {checkpoint_path}")

    # ── Learning curve ─────────────────────────────────────────────────────
    _plot_training_curves(history, save_path=checkpoint_path.parent / "training_curves.png")

    return history


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, str]:
    """Load a saved checkpoint and return (model, arch)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "mobilenet")
    model = build_model(arch=arch, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, arch


def infer_face(
    image_path: str | Path,
    checkpoint_path: str | Path,
    grid: tuple[int, int] = (2, 2),
    confidence_threshold: float = 0.5,
    save_debug_plot: bool = True,
    debug_plot_path: str | Path = "face_debug.png",
) -> list[int]:
    """
    Classify the sticker colors on a single 2×2 cube face.

    The function divides the face image into a ``grid``-shaped grid of sticker
    patches (default 2×2), runs the classifier on each patch, and returns a
    flat list of color integers in reading order (left→right, top→bottom).

    Args:
        image_path:           Path to a cropped photo of one cube face.
        checkpoint_path:      Path to the ``.pt`` file saved by ``train()``.
        grid:                 (rows, cols) sticker grid on this face (2×2 for 2×2 cube).
        confidence_threshold: Warn (and mark in the plot) if max softmax prob < this.
        save_debug_plot:      Whether to write a matplotlib debug image.
        debug_plot_path:      Where to save the debug plot.

    Returns:
        List of ``rows × cols`` integers (0–5), reading order.
        E.g. for a 2×2 face: [top-left, top-right, bottom-left, bottom-right]

    Color encoding:
        0=White  1=Yellow  2=Green  3=Blue  4=Red  5=Orange
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _arch = load_model(checkpoint_path, device)

    transform = get_transforms("val")   # no augmentation at inference time

    # ── Load and split face image ─────────────────────────────────────────
    face_img = Image.open(image_path).convert("RGB")
    W, H = face_img.size
    rows, cols = grid
    patch_w = W // cols
    patch_h = H // rows

    results: list[int]   = []
    confidences: list[float] = []
    patches: list[Image.Image] = []

    for r in range(rows):
        for c in range(cols):
            left   = c * patch_w
            top    = r * patch_h
            right  = left + patch_w
            bottom = top  + patch_h
            patch  = face_img.crop((left, top, right, bottom))
            patches.append(patch)

            tensor = transform(patch).unsqueeze(0).to(device)   # (1, 3, 224, 224)
            with torch.no_grad():
                logits = model(tensor)                           # (1, 6)
                probs  = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)

            color_idx = pred.item()
            conf_val  = conf.item()

            results.append(color_idx)
            confidences.append(conf_val)

            if conf_val < confidence_threshold:
                print(
                    f"[WARN] Sticker ({r},{c}): low confidence {conf_val:.2f} "
                    f"for predicted class '{IDX_TO_CLASS[color_idx]}'"
                )

    print(f"\nFace inference result:")
    for i, (ci, cf) in enumerate(zip(results, confidences)):
        r, c = divmod(i, cols)
        print(f"  Sticker ({r},{c}): {IDX_TO_CLASS[ci]:<8} (conf={cf:.3f})")

    # ── Debug plot ────────────────────────────────────────────────────────
    if save_debug_plot:
        _plot_face_inference(
            face_img=face_img,
            patches=patches,
            predictions=results,
            confidences=confidences,
            grid=grid,
            threshold=confidence_threshold,
            save_path=debug_plot_path,
        )

    return results


def infer_batch(
    image_paths: list[str | Path],
    checkpoint_path: str | Path,
) -> list[int]:
    """
    Classify a list of individual sticker images (one image per sticker).
    Returns a list of color integers in the same order as ``image_paths``.

    Useful when the sticker segmentation is done externally and you just
    need per-sticker classification.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _arch = load_model(checkpoint_path, device)
    transform = get_transforms("val")

    results = []
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = model(batch)
        preds  = logits.argmax(dim=1).tolist()

    return preds


# ─── Misclassification debugger ───────────────────────────────────────────────

def debug_misclassifications(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    split: str = "val",
    max_shown: int = 40,
    save_path: str | Path = "misclassifications.png",
    batch_size: int = 64,
    num_workers: int = 4,
) -> None:
    """
    Run the model over a dataset split and visualise misclassified stickers.

    Produces a grid of images where:
      • The image border is RED for misclassified, GREEN for correct.
      • The title shows "pred → true" in color names.
      • Low-confidence predictions are marked with ⚠.

    Args:
        data_dir:   Root dataset directory (same as used for training).
        checkpoint_path: Path to saved ``.pt`` checkpoint.
        split:      "val" or "train".
        max_shown:  Maximum number of misclassified examples to display.
        save_path:  Where to write the output PNG.
        batch_size: Eval batch size.
        num_workers: DataLoader workers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _arch = load_model(checkpoint_path, device)

    dataset = RubiksDataset(
        Path(data_dir) / split,
        transform=get_transforms("val"),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    # ── Collect misclassified examples ────────────────────────────────────
    misses: list[dict] = []   # {path, pred, true, conf}

    model.eval()
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for img_t, true, pred, conf, path in zip(
                images.cpu(), labels, preds.cpu(), confs.cpu(), paths
            ):
                if pred.item() != true.item():
                    misses.append({
                        "path": path,
                        "pred": pred.item(),
                        "true": true.item(),
                        "conf": conf.item(),
                        "tensor": img_t,
                    })
                    if len(misses) >= max_shown * 4:
                        break

    total_samples   = len(dataset)
    total_correct   = total_samples - len(misses)    # approximate (may exceed max_shown)
    print(f"\nDebug on '{split}' split ({total_samples} samples)")
    print(f"Found {len(misses)} misclassification examples (showing up to {max_shown})")

    if not misses:
        print("No misclassifications found — model is perfect on this split!")
        return

    # ── Per-class confusion summary ───────────────────────────────────────
    confusion = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]
    for m in misses:
        confusion[m["true"]][m["pred"]] += 1

    print("\nConfusion (true \\ pred):")
    header = f"{'':>8}" + "".join(f"{IDX_TO_CLASS[j]:>8}" for j in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = f"{IDX_TO_CLASS[i]:>8}" + "".join(f"{confusion[i][j]:>8}" for j in range(NUM_CLASSES))
        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────
    shown = misses[:max_shown]
    ncols = 8
    nrows = (len(shown) + ncols - 1) // ncols

    # Un-normalize for display
    inv_mean = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225])
    inv_std  = torch.tensor([1/0.229, 1/0.224, 1/0.225])

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.4))
    axes = np.array(axes).reshape(-1)   # flatten for easy indexing

    for ax in axes:
        ax.axis("off")

    for ax, miss in zip(axes, shown):
        img_t = miss["tensor"]
        # Reverse ImageNet normalisation for display
        img_display = img_t.permute(1, 2, 0)                          # CHW → HWC
        img_display = img_display * torch.tensor([0.229, 0.224, 0.225])
        img_display = img_display + torch.tensor([0.485, 0.456, 0.406])
        img_display = img_display.clamp(0, 1).numpy()

        ax.imshow(img_display)

        pred_name = IDX_TO_CLASS[miss["pred"]]
        true_name = IDX_TO_CLASS[miss["true"]]
        conf      = miss["conf"]
        warn      = " ⚠" if conf < 0.5 else ""

        # Color-coded title: red for wrong, swatch colors for labels
        ax.set_title(
            f"P:{pred_name}{warn}\nT:{true_name}  {conf:.2f}",
            fontsize=7,
            color="darkred",
            pad=2,
        )

        # Colored border: predicted color on top, true color on bottom
        for spine in ax.spines.values():
            spine.set_edgecolor(IDX_TO_HEX[miss["pred"]])
            spine.set_linewidth(3)

    # Legend
    legend_patches = [
        mpatches.Patch(color=IDX_TO_HEX[i], label=IDX_TO_CLASS[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=NUM_CLASSES,
        fontsize=9,
        framealpha=0.9,
        title="Colors (border = predicted)",
    )

    fig.suptitle(
        f"Misclassified stickers — {split} split\n"
        f"(P = predicted, T = true, ⚠ = low confidence)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMisclassification plot saved → {save_path}")


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path = "training_curves.png",
) -> None:
    """Plot loss and accuracy curves for train and val splits."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="#CC2200")
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="#0055CC", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training & validation loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    train_acc_pct = [x * 100 for x in history["train_acc"]]
    val_acc_pct   = [x * 100 for x in history["val_acc"]]
    ax2.plot(epochs, train_acc_pct, label="Train acc", color="#CC2200")
    ax2.plot(epochs, val_acc_pct,   label="Val acc",   color="#0055CC", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & validation accuracy")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Mark best val epoch
    best_ep = int(np.argmax(history["val_acc"])) + 1
    ax2.axvline(best_ep, color="green", linestyle=":", alpha=0.7,
                label=f"Best val epoch {best_ep}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


def _plot_face_inference(
    face_img: Image.Image,
    patches: list[Image.Image],
    predictions: list[int],
    confidences: list[float],
    grid: tuple[int, int],
    threshold: float,
    save_path: str | Path,
) -> None:
    """
    Visualise the face inference result:
      Left panel  — the original face with a colored grid overlay.
      Right panel — each sticker patch with predicted color and confidence.
    """
    rows, cols = grid
    n_patches = rows * cols

    fig, axes = plt.subplots(
        rows, cols + 1,
        figsize=((cols + 1) * 2.2, rows * 2.2),
        gridspec_kw={"width_ratios": [cols] + [1] * cols},
    )
    # Use first column (spanning rows) for the face overview
    # Matplotlib doesn't span natively in subplots — use gridspec instead
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    gs  = fig.add_gridspec(rows, cols + 2, hspace=0.4, wspace=0.3)

    # Face overview (left panel, spans all rows)
    ax_face = fig.add_subplot(gs[:, :2])
    draw = ImageDraw.Draw(face_img.copy())
    W, H = face_img.size
    pw, ph = W // cols, H // rows
    ax_face.imshow(face_img)
    ax_face.set_title("Input face", fontsize=10)
    ax_face.axis("off")

    # Draw predicted-color grid on face overview
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        r, c = divmod(i, cols)
        x0, y0 = c * pw, r * ph
        rect = plt.Rectangle(
            (x0, y0), pw, ph,
            linewidth=3,
            edgecolor=IDX_TO_HEX[pred],
            facecolor=IDX_TO_HEX[pred],
            alpha=0.35,
        )
        ax_face.add_patch(rect)
        ax_face.text(
            x0 + pw/2, y0 + ph/2,
            f"{IDX_TO_CLASS[pred][0]}\n{conf:.2f}",
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="black",
        )

    # Individual sticker patches (right panels)
    for i, (patch, pred, conf) in enumerate(zip(patches, predictions, confidences)):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c + 2])
        ax.imshow(patch)
        low_conf = conf < threshold
        title_color = "darkred" if low_conf else "black"
        ax.set_title(
            f"{IDX_TO_CLASS[pred]}\n{conf:.2f}{'  ⚠' if low_conf else ''}",
            fontsize=7, color=title_color, pad=2,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(IDX_TO_HEX[pred])
            spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

    # Color legend
    legend_patches = [
        mpatches.Patch(color=IDX_TO_HEX[i], label=IDX_TO_CLASS[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=NUM_CLASSES,
               fontsize=8, framealpha=0.9)

    fig.suptitle("2×2 Face — sticker color inference", fontsize=12, y=1.02)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Inference debug plot saved → {save_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rubik's cube sticker color classifier — train / infer / debug",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    t = sub.add_parser("train", help="Fine-tune a pre-trained model on the Kaggle dataset")
    t.add_argument("--data_dir",       default="./data",          help="Root dataset directory")
    t.add_argument("--checkpoint",     default="best_model.pt",   help="Where to save the best model")
    t.add_argument("--model",          default="mobilenet",
                   choices=["mobilenet", "resnet18", "resnet50"],  help="Backbone architecture")
    t.add_argument("--epochs",         type=int,   default=20,    help="Total training epochs")
    t.add_argument("--batch_size",     type=int,   default=32)
    t.add_argument("--lr_head",        type=float, default=1e-3,  help="LR for head-only phase")
    t.add_argument("--lr_backbone",    type=float, default=1e-4,  help="LR for fine-tuning phase")
    t.add_argument("--head_epochs",    type=int,   default=3,     help="Epochs to train head-only")
    t.add_argument("--num_workers",    type=int,   default=4)
    t.add_argument("--dropout",        type=float, default=0.3)

    # ── infer ──────────────────────────────────────────────────────────────
    i = sub.add_parser("infer", help="Classify sticker colors on a 2×2 face image")
    i.add_argument("--image",          required=True,             help="Path to face image")
    i.add_argument("--checkpoint",     default="best_model.pt")
    i.add_argument("--rows",           type=int,   default=2,     help="Sticker rows on face")
    i.add_argument("--cols",           type=int,   default=2,     help="Sticker columns on face")
    i.add_argument("--threshold",      type=float, default=0.5,   help="Low-confidence warning threshold")
    i.add_argument("--debug_plot",     default="face_debug.png",  help="Where to save debug plot")

    # ── debug ──────────────────────────────────────────────────────────────
    d = sub.add_parser("debug", help="Visualise misclassified stickers from the val set")
    d.add_argument("--data_dir",       default="./data")
    d.add_argument("--checkpoint",     default="best_model.pt")
    d.add_argument("--split",          default="val",  choices=["train", "val"])
    d.add_argument("--max_shown",      type=int,   default=40)
    d.add_argument("--save_path",      default="misclassifications.png")
    d.add_argument("--batch_size",     type=int,   default=64)
    d.add_argument("--num_workers",    type=int,   default=4)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        history = train(
            data_dir        = args.data_dir,
            checkpoint_path = args.checkpoint,
            arch            = args.model,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            lr_head         = args.lr_head,
            lr_backbone     = args.lr_backbone,
            head_only_epochs= args.head_epochs,
            num_workers     = args.num_workers,
            dropout         = args.dropout,
        )
        # Save history for later inspection
        history_path = Path(args.checkpoint).with_suffix(".json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History saved → {history_path}")

    elif args.mode == "infer":
        colors = infer_face(
            image_path              = args.image,
            checkpoint_path         = args.checkpoint,
            grid                    = (args.rows, args.cols),
            confidence_threshold    = args.threshold,
            save_debug_plot         = True,
            debug_plot_path         = args.debug_plot,
        )
        print(f"\nReturned color array: {colors}")
        print("(Paste this into your FastAPI /solve endpoint as part of the 24-element state)")

    elif args.mode == "debug":
        debug_misclassifications(
            data_dir        = args.data_dir,
            checkpoint_path = args.checkpoint,
            split           = args.split,
            max_shown       = args.max_shown,
            save_path       = args.save_path,
            batch_size      = args.batch_size,
            num_workers     = args.num_workers,
        )


if __name__ == "__main__":
    main()
