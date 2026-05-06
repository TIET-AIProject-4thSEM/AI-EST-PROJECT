"""
================================================================================
TATA STEEL — Binary Surface Crack Detector
crack_detector.py
================================================================================

ARCHITECTURE:
  Backbone   : EfficientNet-B3 (ImageNet pre-trained) — replaceable via --arch
  Head       : GlobalAvgPool → Dropout → BN → FC(256) → Dropout → FC(2)
  Loss       : Label-smoothed CrossEntropy  (+ optional focal loss)
  Optimiser  : AdamW with decoupled weight decay
  Scheduler  : Cosine annealing with linear warm-up
  Precision  : Automatic Mixed Precision (AMP, torch.cuda.amp)

TRAINING STRATEGY (three phases):
  Phase 1 – HEAD ONLY        : freeze all backbone weights, train head 5 epochs
  Phase 2 – TOP UNFREEZE     : unfreeze top 30% of backbone, lower LR, 10 epochs
  Phase 3 – FULL FINE-TUNE   : unfreeze entire network, very low LR, rest of epochs

AUGMENTATION (Albumentations):
  Train  : random flip/rotate, brightness/contrast, CLAHE, Gaussian noise,
           elastic distort, coarse dropout (CutOut), normalise → tensor
  Val/Test: only normalise → tensor

EXPLAINABLE AI:
  Grad-CAM heatmaps are generated after training on a random batch of
  test images (both classes) and saved as a grid PNG.

METRICS:
  Per epoch  : Accuracy, Precision, Recall, F1, AUC-ROC, Avg Loss
  Final test : Full classification report + confusion matrix + ROC curve
  TensorBoard: All metrics + sample images logged live

OUTPUTS:
  outputs/
    best_model.pth          ← best checkpoint by val F1
    last_model.pth          ← final epoch checkpoint
    training_curves.png
    confusion_matrix.png
    roc_curve.png
    gradcam_grid.png
    metrics_history.json
    model_card.txt

USAGE:
  python crack_detector.py --data_dir data/steel_binary

  # Full options:
  python crack_detector.py \\
      --data_dir   data/steel_binary \\
      --output_dir outputs \\
      --arch       efficientnet_b3 \\
      --epochs     40 \\
      --batch_size 32 \\
      --lr         3e-4 \\
      --workers    4 \\
      --seed       42
================================================================================
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ────────────────────────────────────────────────────────────
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as tv_models
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crack_detector")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE  = 224
MEAN      = [0.485, 0.456, 0.406]   # ImageNet stats
STD       = [0.229, 0.224, 0.225]
CLASS_MAP = {0: "NO_CRACK", 1: "CRACK"}  # index → name

# Supported backbones → (constructor, feature_dim)
ARCH_REGISTRY: Dict[str, Tuple] = {
    "efficientnet_b0": (tv_models.efficientnet_b0, 1280),
    "efficientnet_b3": (tv_models.efficientnet_b3, 1536),
    "efficientnet_b5": (tv_models.efficientnet_b5, 2048),
    "resnet50":        (tv_models.resnet50,         2048),
    "resnet101":       (tv_models.resnet101,        2048),
    "convnext_small":  (tv_models.convnext_small,   768),
    "convnext_base":   (tv_models.convnext_base,    1024),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_transforms(split: str) -> A.Compose:
    """
    Return Albumentations pipelines.

    Train:  heavy augmentation targeting steel texture variance
             — random rotations cover different orientations of cracks
             — CLAHE boosts micro-contrast to make micro-cracks visible
             — elastic/grid distort simulates slight imaging misalignment
             — coarse dropout acts as CutOut regularisation

    Val/Test: only standardise + normalise (no randomness)
    """
    if split == "train":
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            # ─ geometric ─────────────────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=20, border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),
            # ─ photometric ───────────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.6
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20,
                val_shift_limit=10, p=0.3
            ),
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.7, 1.3), p=0.3),
            A.GaussNoise(var_limit=(10, 60), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # ─ structural ────────────────────────────────────────────────
            A.ElasticTransform(
                alpha=80, sigma=10,
                border_mode=cv2.BORDER_REFLECT, p=0.3
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            # ─ regularisation ────────────────────────────────────────────
            A.CoarseDropout(
                max_holes=8, max_height=24, max_width=24,
                min_holes=1, fill_value=0, p=0.4
            ),
            # ─ normalise → tensor ─────────────────────────────────────────
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])


class SteelDataset(Dataset):
    """
    Reads images from:
      root/<split>/CRACK/     → label 1
      root/<split>/NO_CRACK/  → label 0

    Images are loaded as RGB numpy arrays and passed through
    an Albumentations transform pipeline.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: Path, split: str):
        self.transform = build_transforms(split)
        self.samples: List[Tuple[Path, int]] = []

        for label_name, label_idx in [("CRACK", 1), ("NO_CRACK", 0)]:
            folder = root / split / label_name
            if not folder.exists():
                log.warning(f"  Missing: {folder}")
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((p, label_idx))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}/{split}/")

        self.class_counts = {0: 0, 1: 0}
        for _, lbl in self.samples:
            self.class_counts[lbl] += 1

        log.info(
            f"  [{split:5}] {len(self.samples):>5} images  "
            f"| CRACK={self.class_counts[1]}  NO_CRACK={self.class_counts[0]}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Fallback: try PIL
            img = np.array(Image.open(path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.transform(image=img)
        return result["image"].float(), label


def make_weighted_sampler(dataset: SteelDataset) -> WeightedRandomSampler:
    """
    Over-sample the minority class so every training batch is class-balanced,
    even if the dataset becomes slightly imbalanced after augmentation.
    """
    n_total = len(dataset)
    weights = []
    class_weight = {
        k: n_total / (2.0 * v)
        for k, v in dataset.class_counts.items()
    }
    for _, label in dataset.samples:
        weights.append(class_weight[label])
    return WeightedRandomSampler(
        weights=weights, num_samples=n_total, replacement=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL
# ══════════════════════════════════════════════════════════════════════════════

class CrackClassifier(nn.Module):
    """
    Transfer-learning wrapper around any torchvision backbone.

    Head:
      GlobalAveragePool → Dropout(p1) → BN → Linear(feat→256)
      → ReLU → Dropout(p2) → Linear(256→2)

    Two drop rates let us dial regularisation without touching the backbone.
    """

    def __init__(
        self,
        arch: str = "efficientnet_b3",
        num_classes: int = 2,
        drop1: float = 0.4,
        drop2: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.arch = arch

        if arch not in ARCH_REGISTRY:
            raise ValueError(
                f"Unknown arch '{arch}'. Choose from: {list(ARCH_REGISTRY)}"
            )
        constructor, feat_dim = ARCH_REGISTRY[arch]
        weights = "IMAGENET1K_V1" if pretrained else None
        base    = constructor(weights=weights)

        # ── strip original classification head ───────────────────────────
        if arch.startswith("efficientnet"):
            self.backbone = base.features
        elif arch.startswith("resnet"):
            self.backbone = nn.Sequential(*list(base.children())[:-2])
        elif arch.startswith("convnext"):
            self.backbone = base.features
        else:
            raise ValueError(f"Unsupported arch family: {arch}")

        # ── custom classification head ────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=drop1),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop2),
            nn.Linear(256, num_classes),
        )
        self._feat_dim = feat_dim

        # ── Grad-CAM hook storage ─────────────────────────────────────────
        self._gradcam_grads: Optional[torch.Tensor] = None
        self._gradcam_acts:  Optional[torch.Tensor] = None
        self._hook_handles   = []

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # (B, C, H, W)
        pooled = self.pool(feat)         # (B, C, 1, 1)
        flat   = pooled.view(pooled.size(0), -1)
        return self.head(flat)

    # ── phase helpers ────────────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_fraction(self, fraction: float = 0.3) -> None:
        """Unfreeze the last `fraction` of backbone layers."""
        params = list(self.backbone.parameters())
        n_unfreeze = max(1, int(len(params) * fraction))
        for p in params:
            p.requires_grad = False
        for p in params[-n_unfreeze:]:
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Grad-CAM hooks ───────────────────────────────────────────────────
    def _find_last_conv(self) -> nn.Module:
        """Return the final convolutional layer of the backbone."""
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d found in backbone for Grad-CAM.")
        return last_conv

    def register_gradcam_hooks(self) -> None:
        """Register forward + backward hooks on the last conv layer."""
        self.remove_gradcam_hooks()
        target = self._find_last_conv()

        def save_activation(_, __, output):
            self._gradcam_acts = output.detach()

        def save_gradient(_, grad_in, grad_out):
            self._gradcam_grads = grad_out[0].detach()

        self._hook_handles.append(
            target.register_forward_hook(save_activation)
        )
        self._hook_handles.append(
            target.register_full_backward_hook(save_gradient)
        )

    def remove_gradcam_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def gradcam(
        self, x: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image tensor (1,C,H,W).
        Returns a (H,W) numpy float32 array in [0,1].
        """
        self.eval()
        self.register_gradcam_hooks()
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x.requires_grad_(True)

        logits = self.forward(x)           # (1, 2)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.zero_grad()
        logits[0, class_idx].backward()

        grads = self._gradcam_grads        # (1, C, H, W)
        acts  = self._gradcam_acts         # (1, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # global avg pool
        cam     = (weights * acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        self.remove_gradcam_hooks()
        return cam


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LOSS
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing (Szegedy et al. 2016).
    Smoothing ε > 0 prevents the network from becoming overconfident
    on noisy or ambiguous boundary labels.
    """

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_cls    = pred.size(-1)
        log_prob = F.log_softmax(pred, dim=-1)

        # one-hot smoothed target
        with torch.no_grad():
            smooth_val = self.smoothing / n_cls
            true_val   = 1.0 - self.smoothing + smooth_val
            soft_target = torch.full_like(log_prob, smooth_val)
            soft_target.scatter_(1, target.unsqueeze(1), true_val)

        loss = -(soft_target * log_prob)

        if self.weight is not None:
            w = self.weight[target].unsqueeze(1)
            loss = loss * w

        return loss.sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al. 2017) — down-weights easy negatives,
    so the model focuses training on hard-to-classify boundary examples.
    Used as an optional alternative / additive term.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(pred, dim=-1)
        p     = torch.exp(log_p)
        focal = (1 - p) ** self.gamma * log_p
        return F.nll_loss(
            -focal.gather(1, target.unsqueeze(1)).squeeze(1),
            target, weight=self.weight
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SCHEDULER  (cosine annealing + linear warm-up)
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warm-up for `warmup_epochs` epochs, then cosine decay
    to `min_lr_ratio * base_lr` over the remaining epochs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.05,
    ):
        self.warmup = warmup_epochs
        self.total  = total_epochs
        self.min_r  = min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            if epoch < self.warmup:
                return (epoch + 1) / max(1, self.warmup)
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            cosine   = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_r + (1 - self.min_r) * cosine

        super().__init__(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 8, delta: float = 1e-4):
        self.patience  = patience
        self.delta     = delta
        self.best      = -np.inf
        self.counter   = 0
        self.triggered = False

    def __call__(self, score: float) -> bool:
        if score > self.best + self.delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """Run one full pass over loader, return dict of metrics."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    n     = len(all_labels)
    lbl   = np.array(all_labels)
    pred  = np.array(all_preds)
    prob  = np.array(all_probs)

    tp = ((pred == 1) & (lbl == 1)).sum()
    fp = ((pred == 1) & (lbl == 0)).sum()
    fn = ((pred == 0) & (lbl == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "loss":      total_loss / n,
        "accuracy":  (pred == lbl).mean(),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "auc":       roc_auc_score(lbl, prob) if len(np.unique(lbl)) > 1 else 0.0,
        "labels":    lbl,
        "probs":     prob,
        "preds":     pred,
    }


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    grad_clip: float = 1.0,
) -> Dict:
    """Train for one epoch, return dict of training metrics."""
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.detach().argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    n    = len(all_labels)
    lbl  = np.array(all_labels)
    pred = np.array(all_preds)
    tp   = ((pred == 1) & (lbl == 1)).sum()
    fp   = ((pred == 1) & (lbl == 0)).sum()
    fn   = ((pred == 0) & (lbl == 1)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        "loss":      total_loss / n,
        "accuracy":  (pred == lbl).mean(),
        "f1":        float(f1),
    }


def get_phase(epoch: int, phase1_end: int, phase2_end: int) -> int:
    """Return training phase (1, 2, or 3) given the current epoch."""
    if epoch < phase1_end:
        return 1
    if epoch < phase2_end:
        return 2
    return 3


# ══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: Dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Curves — Tata Steel Crack Detector", fontsize=14, fontweight="bold")

    def plot(ax, key_tr, key_va, title, ylabel):
        ax.plot(epochs, history[key_tr], label="Train", linewidth=2, color="#2196F3")
        ax.plot(epochs, history[key_va], label="Val",   linewidth=2, color="#F44336")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    plot(axes[0][0], "train_loss",     "val_loss",     "Loss",     "CE Loss")
    plot(axes[0][1], "train_accuracy", "val_accuracy", "Accuracy", "Accuracy")
    plot(axes[1][0], "train_f1",       "val_f1",       "F1 Score", "F1")
    axes[1][1].plot(epochs, history["val_auc"], linewidth=2, color="#4CAF50", label="Val AUC")
    axes[1][1].set_title("AUC-ROC"); axes[1][1].set_xlabel("Epoch")
    axes[1][1].set_ylabel("AUC"); axes[1][1].legend(); axes[1][1].grid(True, alpha=0.3)

    # Mark best epoch
    best_ep = int(np.argmax(history["val_f1"])) + 1
    for ax in axes.flat:
        ax.axvline(best_ep, linestyle="--", color="gray", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, out_path: Path) -> None:
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NO_CRACK", "CRACK"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2.5, color="#1565C0",
            label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def unnormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised CHW tensor to HWC uint8 RGB for visualisation."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def generate_gradcam_grid(
    model:      CrackClassifier,
    test_loader: DataLoader,
    device:      torch.device,
    out_path:    Path,
    n_cols: int = 4,
) -> None:
    """
    Draw a grid of n_rows × n_cols panels, each showing:
      Left:  original image
      Right: Grad-CAM heatmap overlaid (jet colourmap)

    Rows: 2 CRACK examples  +  2 NO_CRACK examples
    Cols: n_cols images each
    """
    model.eval()

    # collect samples by class
    crack_samples:    List[torch.Tensor] = []
    no_crack_samples: List[torch.Tensor] = []
    need = n_cols

    for imgs, labels in test_loader:
        for img, lbl in zip(imgs, labels):
            if lbl.item() == 1 and len(crack_samples)    < need:
                crack_samples.append(img)
            if lbl.item() == 0 and len(no_crack_samples) < need:
                no_crack_samples.append(img)
        if len(crack_samples) >= need and len(no_crack_samples) >= need:
            break

    all_rows = [
        ("CRACK",    crack_samples[:need]),
        ("NO_CRACK", no_crack_samples[:need]),
    ]

    n_rows = len(all_rows)
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(
        "Grad-CAM Explainability — Tata Steel Crack Detector\n"
        "Original (left) | Heatmap overlay (right)",
        fontsize=13, fontweight="bold"
    )
    gs   = gridspec.GridSpec(n_rows, n_cols * 2, figure=fig,
                              hspace=0.35, wspace=0.05)

    for row_idx, (class_name, samples) in enumerate(all_rows):
        for col_idx, img_t in enumerate(samples):
            img_np = unnormalise(img_t)

            # compute Grad-CAM
            cam = model.gradcam(img_t.to(device), class_idx=None)
            cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
            heatmap_rgb = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_np, 0.55, heatmap_rgb, 0.45, 0)

            # original
            ax_orig = fig.add_subplot(gs[row_idx, col_idx * 2])
            ax_orig.imshow(img_np)
            ax_orig.axis("off")
            if col_idx == 0:
                ax_orig.set_ylabel(
                    class_name, fontsize=11, fontweight="bold",
                    rotation=0, labelpad=65, va="center"
                )

            # overlay
            ax_cam = fig.add_subplot(gs[row_idx, col_idx * 2 + 1])
            ax_cam.imshow(overlay)
            ax_cam.axis("off")

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"  Grad-CAM grid saved → {out_path}")


def write_model_card(
    args:    argparse.Namespace,
    history: Dict,
    test_metrics: Dict,
    out_path: Path,
) -> None:
    best_f1  = max(history["val_f1"])
    best_ep  = int(np.argmax(history["val_f1"])) + 1
    lines = [
        "=" * 60,
        "  TATA STEEL — Crack Detector  |  Model Card",
        "=" * 60,
        "",
        f"  Architecture   : {args.arch}",
        f"  Image size     : {IMG_SIZE}×{IMG_SIZE}",
        f"  Batch size     : {args.batch_size}",
        f"  Epochs trained : {len(history['train_loss'])}",
        f"  Best epoch     : {best_ep}  (by val F1)",
        f"  Seed           : {args.seed}",
        "",
        "  ── Validation (best epoch) ──────────────────────",
        f"  F1       : {best_f1:.4f}",
        f"  AUC-ROC  : {history['val_auc'][best_ep-1]:.4f}",
        f"  Accuracy : {history['val_accuracy'][best_ep-1]:.4f}",
        "",
        "  ── Test set ────────────────────────────────────",
        f"  Loss     : {test_metrics['loss']:.4f}",
        f"  Accuracy : {test_metrics['accuracy']:.4f}",
        f"  Precision: {test_metrics['precision']:.4f}",
        f"  Recall   : {test_metrics['recall']:.4f}",
        f"  F1       : {test_metrics['f1']:.4f}",
        f"  AUC-ROC  : {test_metrics['auc']:.4f}",
        "",
        "  ── Data ─────────────────────────────────────────",
        f"  Train dir: {args.data_dir}",
        "",
        "  ── Training strategy ────────────────────────────",
        "  Phase 1 : head-only (backbone frozen)",
        "  Phase 2 : top 30% of backbone unfrozen",
        "  Phase 3 : full network fine-tuning",
        "",
        "=" * 60,
    ]
    out_path.write_text("\n".join(lines))
    log.info(f"  Model card → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  INFERENCE (production helper)
# ══════════════════════════════════════════════════════════════════════════════

class CrackInferencer:
    """
    Lightweight wrapper for production inference.

    Usage:
        infer = CrackInferencer("outputs/best_model.pth", arch="efficientnet_b3")
        label, confidence, cam = infer.predict("steel_plate.jpg")
    """

    def __init__(self, checkpoint_path: str, arch: str = "efficientnet_b3",
                 device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = CrackClassifier(arch=arch, pretrained=False)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.transform = build_transforms("test")

    @torch.no_grad()
    def predict(
        self, image_path: str, return_gradcam: bool = False
    ) -> Tuple[str, float, Optional[np.ndarray]]:
        """
        Returns:
          label      : "CRACK" or "NO_CRACK"
          confidence : probability of the predicted class
          cam        : Grad-CAM heatmap (HxW float32 [0,1]) or None
        """
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image=img)["image"].float()
        inp    = tensor.unsqueeze(0).to(self.device)

        logits = self.model(inp)
        probs  = F.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
        conf   = probs[idx].item()
        label  = CLASS_MAP[idx]

        cam = None
        if return_gradcam:
            cam = self.model.gradcam(tensor.to(self.device), class_idx=idx)
            cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

        return label, conf, cam


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main() -> None:
    # ── CLI ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Tata Steel — Binary Crack Detector Training"
    )
    parser.add_argument("--data_dir",   type=str, default="data/steel_binary")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--arch",       type=str, default="efficientnet_b3",
                        choices=list(ARCH_REGISTRY))
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early stopping patience epochs")
    parser.add_argument("--phase1_epochs", type=int, default=5,
                        help="Head-only training epochs")
    parser.add_argument("--phase2_epochs", type=int, default=15,
                        help="Top-fraction unfreeze epochs (cumulative)")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_amp",     action="store_true",
                        help="Disable AMP (automatic mixed precision)")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and torch.cuda.is_available()

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TATA STEEL — Surface Crack Detector")
    print("=" * 60)
    print(f"  Arch        : {args.arch}")
    print(f"  Device      : {device}  |  AMP: {use_amp}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Output dir  : {out_dir}")
    print("=" * 60 + "\n")

    # ── Datasets & Loaders ───────────────────────────────────────────────
    data_root = Path(args.data_dir)
    log.info("Loading datasets …")
    train_ds = SteelDataset(data_root, "train")
    val_ds   = SteelDataset(data_root, "val")
    test_ds  = SteelDataset(data_root, "test")

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    log.info(f"Building model: {args.arch} …")
    model = CrackClassifier(arch=args.arch).to(device)
    model.freeze_backbone()
    log.info(f"  Phase 1 — head only: {model.trainable_params():,} trainable params")

    # ── Loss ─────────────────────────────────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Head params: higher LR;  backbone: lower LR (set in phase transitions)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
    )
    scaler = GradScaler(enabled=use_amp)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    # Log a sample batch of train images
    sample_imgs, sample_labels = next(iter(train_loader))
    grid_imgs = sample_imgs[:8]
    # unnormalise for TB display
    unnorm = torch.stack([
        torch.from_numpy(unnormalise(g).transpose(2, 0, 1)) for g in grid_imgs
    ])
    writer.add_images("train/sample_batch", unnorm, global_step=0)

    # ── Training history ──────────────────────────────────────────────────
    history = {k: [] for k in [
        "train_loss", "train_accuracy", "train_f1",
        "val_loss",   "val_accuracy",   "val_f1",   "val_auc",
        "lr",
    ]}

    best_val_f1   = -np.inf
    early_stopper = EarlyStopping(patience=args.patience)
    current_phase = 0   # track so we only log phase changes once

    phase1_end = args.phase1_epochs
    phase2_end = args.phase2_epochs

    log.info("Starting training …\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Phase transitions ─────────────────────────────────────────────
        phase = get_phase(epoch - 1, phase1_end, phase2_end)

        if phase != current_phase:
            current_phase = phase
            if phase == 2:
                log.info("  ⇒ Phase 2: unfreeze top 30% of backbone")
                model.unfreeze_top_fraction(0.30)
                # rebuild optimiser with separate backbone / head LR groups
                optimizer = torch.optim.AdamW([
                    {"params": [p for p in model.head.parameters()],
                     "lr": args.lr},
                    {"params": [p for p in model.backbone.parameters()
                                if p.requires_grad],
                     "lr": args.lr * 0.1},
                ], weight_decay=args.weight_decay)
                scheduler = WarmupCosineScheduler(
                    optimizer,
                    warmup_epochs=0,
                    total_epochs=args.epochs - phase1_end,
                    min_lr_ratio=0.01,
                )
                log.info(f"  Trainable params: {model.trainable_params():,}")

            elif phase == 3:
                log.info("  ⇒ Phase 3: full network fine-tuning")
                model.unfreeze_all()
                optimizer = torch.optim.AdamW([
                    {"params": [p for p in model.head.parameters()],
                     "lr": args.lr * 0.1},
                    {"params": [p for p in model.backbone.parameters()],
                     "lr": args.lr * 0.01},
                ], weight_decay=args.weight_decay)
                scheduler = WarmupCosineScheduler(
                    optimizer,
                    warmup_epochs=0,
                    total_epochs=args.epochs - phase2_end,
                    min_lr_ratio=0.005,
                )
                log.info(f"  Trainable params: {model.trainable_params():,}")

        # ── Train ────────────────────────────────────────────────────────
        tr_metrics  = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────────
        history["train_loss"].append(tr_metrics["loss"])
        history["train_accuracy"].append(tr_metrics["accuracy"])
        history["train_f1"].append(tr_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        history["lr"].append(current_lr)

        for tag, val in [
            ("Loss/train",       tr_metrics["loss"]),
            ("Loss/val",         val_metrics["loss"]),
            ("Accuracy/train",   tr_metrics["accuracy"]),
            ("Accuracy/val",     val_metrics["accuracy"]),
            ("F1/train",         tr_metrics["f1"]),
            ("F1/val",           val_metrics["f1"]),
            ("AUC/val",          val_metrics["auc"]),
            ("LR",               current_lr),
        ]:
            writer.add_scalar(tag, val, epoch)

        log.info(
            f"  Ep {epoch:03d}/{args.epochs}  Ph{phase}  "
            f"[{elapsed:.0f}s]  "
            f"Loss {tr_metrics['loss']:.4f}/{val_metrics['loss']:.4f}  "
            f"Acc {tr_metrics['accuracy']:.3f}/{val_metrics['accuracy']:.3f}  "
            f"F1 {tr_metrics['f1']:.3f}/{val_metrics['f1']:.3f}  "
            f"AUC {val_metrics['auc']:.3f}  "
            f"LR {current_lr:.2e}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────
        checkpoint = {
            "model":  model.state_dict(),
            "optim":  optimizer.state_dict(),
            "epoch":  epoch,
            "arch":   args.arch,
            "val_f1": val_metrics["f1"],
        }
        torch.save(checkpoint, out_dir / "last_model.pth")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(checkpoint, out_dir / "best_model.pth")
            log.info(f"  ★ New best val F1 = {best_val_f1:.4f}")

        # ── Early stopping ────────────────────────────────────────────────
        if early_stopper(val_metrics["f1"]):
            log.info(
                f"  Early stopping triggered (no improvement "
                f"for {args.patience} epochs)."
            )
            break

    writer.close()

    # ── Test evaluation ──────────────────────────────────────────────────
    log.info("\nLoading best model for final test evaluation …")
    best_ckpt = torch.load(
        out_dir / "best_model.pth",
        map_location=device,
        weights_only=False  # IMPORTANT
    )   
    model.load_state_dict(best_ckpt["model"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 55)
    print("  FINAL TEST SET RESULTS")
    print("=" * 55)
    report = classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=["NO_CRACK", "CRACK"]
    )
    print(report)
    print(f"  AUC-ROC : {test_metrics['auc']:.4f}")
    print("=" * 55 + "\n")

    # ── Save plots ────────────────────────────────────────────────────────
    log.info("Saving plots …")
    plot_training_curves(history, out_dir / "training_curves.png")
    plot_confusion_matrix(test_metrics["labels"],
                          test_metrics["preds"],
                          out_dir / "confusion_matrix.png")
    plot_roc_curve(test_metrics["labels"],
                   test_metrics["probs"],
                   out_dir / "roc_curve.png")

    # ── Grad-CAM grid ─────────────────────────────────────────────────────
    log.info("Generating Grad-CAM explainability grid …")
    generate_gradcam_grid(
        model, test_loader, device, out_dir / "gradcam_grid.png", n_cols=4
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    with open(out_dir / "metrics_history.json", "w") as f:
        # numpy floats → python floats
        hist_serialisable = {
            k: [float(v) for v in vals]
            for k, vals in history.items()
        }
        json.dump(hist_serialisable, f, indent=2)

    write_model_card(args, history, test_metrics, out_dir / "model_card.txt")

    print("\n" + "=" * 55)
    print("  DONE — all artefacts saved to:", out_dir.resolve())
    print("=" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
