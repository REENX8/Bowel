"""
Simple 2.5D classification pipeline for bowel injury detection from processed
CT volumes.  This script defines a PyTorch dataset that loads 3D volumes
saved as ``.npy`` files, converts them into 2.5D sequences (adjacent slice
triplets), and trains a convolutional neural network followed by a recurrent
network to classify whether a case has bowel injury.

This serves as a lightweight example inspired by the winning solution of
Team Oxygen in the RSNA 2023 Abdominal Trauma Detection challenge.  Their
approach used 3D segmentation, 2.5D CNNs, recurrent layers, auxiliary
segmentation loss and ensembling【327782157900899†L55-L72】.  Here we simplify
significantly:

* **Bowel injury only:** we use labels where ``1`` indicates the presence of
  bowel injury and ``0`` indicates absence.
* **2.5D representation:** each 3D volume of shape ``(Z, H, W)`` is divided
  into ``T`` equidistant time steps (default 32).  For each time step we
  take 3 adjacent slices as the RGB channels for a 2D CNN.  This helps
  incorporate some context along the z‑axis without full 3D convolutions.
* **CNN + GRU:** we use a pre-trained ResNet18 (from torchvision) as the
  feature extractor for each 2D frame, then feed the sequence of features
  into a GRU to aggregate temporal information and produce a final
  classification.
* **GroupKFold:** dataset splitting is done by patient/study name so that
  slices from the same patient do not leak across folds.

To run this script:

1. Install dependencies: ``pip install torch torchvision pandas scikit-learn``
2. Place your processed dataset (e.g. 96×256×256 float16 volumes) in
   ``--data-dir``.  There should be a ``labels.csv`` file in the same
   directory with two columns: ``patient_id`` (matching the file names
   without extension) and ``bowel_injury`` (0 or 1).
3. Run training: ``python bowel_injury_model.py --data-dir /path/to/processed_data``.

The script saves the best model weights and logs basic metrics.  You can
adapt the architecture, hyperparameters, or add augmentation as needed.
"""

from __future__ import annotations

import os
import csv
import math
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


class BowelInjuryDataset(Dataset):
    """Dataset for loading processed CT volumes and converting to 2.5D sequences.

    Each sample corresponds to one case (patient/study).  The volume is
    loaded from an ``.npy`` file of shape (Z, H, W).  We divide the Z
    dimension into ``num_steps`` segments and, for each segment, take
    ``num_slices_per_step`` adjacent slices to form a 3‑channel image.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``path`` (path to ``.npy`` file), ``label`` and
        ``patient_id``.
    num_steps : int
        Number of time steps per case (e.g. 32).
    num_slices_per_step : int
        Number of adjacent slices combined into one frame (e.g. 3).
    transform : callable, optional
        Optional transform applied to each frame (e.g. resizing, normalization).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_steps: int = 32,
        num_slices_per_step: int = 3,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.num_steps = num_steps
        self.num_slices_per_step = num_slices_per_step
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _volume_to_sequence(self, volume: np.ndarray) -> torch.Tensor:
        """Convert a volume into a sequence of 3‑channel frames.

        Parameters
        ----------
        volume : np.ndarray
            3D array of shape (Z, H, W).

        Returns
        -------
        seq : torch.Tensor
            Tensor of shape (num_steps, 3, H, W) ready for CNN.
        """
        z, h, w = volume.shape
        # Determine indices for the center slice of each step
        # We take evenly spaced indices between [0, z - 1]
        centers = np.linspace(0, z - 1, self.num_steps, dtype=int)
        frames = []
        half = self.num_slices_per_step // 2
        for c in centers:
            # Determine start and end slice indices for the window
            start = int(max(c - half, 0))
            end = int(min(c + half + 1, z))
            # Extract slices; if not enough, pad with edge slices
            slc = volume[start:end]
            # Pad if needed to ensure correct number of slices
            if slc.shape[0] < self.num_slices_per_step:
                pad_pre = max(0, half - (c - start))
                pad_post = max(0, (c + half) - (end - 1))
                pre_pad = np.repeat(volume[[start]], pad_pre, axis=0) if pad_pre > 0 else np.empty((0, h, w))
                post_pad = np.repeat(volume[[end - 1]], pad_post, axis=0) if pad_post > 0 else np.empty((0, h, w))
                slc = np.concatenate([pre_pad, slc, post_pad], axis=0)
            # Now slc has shape (num_slices_per_step, H, W)
            if slc.shape[0] != self.num_slices_per_step:
                # In case of edge cases
                slc = np.resize(slc, (self.num_slices_per_step, h, w))
            # Stack into channels
            frame = np.stack([slc[i] for i in range(self.num_slices_per_step)], axis=0)  # shape (3, H, W)
            frames.append(frame)
        seq = np.stack(frames, axis=0)  # shape (num_steps, 3, H, W)
        return torch.from_numpy(seq.astype(np.float32))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        volume = np.load(row['path'])  # shape (Z, H, W)
        seq = self._volume_to_sequence(volume)
        if self.transform:
            # Apply transform to each frame individually
            seq = torch.stack([self.transform(img) for img in seq])
        label = int(row['bowel_injury'])
        return seq, label


class CNNGRUClassifier(nn.Module):
    """Simple 2.5D classifier consisting of a 2D CNN encoder and a GRU."""

    def __init__(self, cnn_name: str = 'resnet18', hidden_size: int = 256, num_classes: int = 1) -> None:
        super().__init__()
        if cnn_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Modify input conv layer to accept 3 channels (already the case)
            # Remove the final fully connected layer
            self.cnn = nn.Sequential(*(list(model.children())[:-1]))  # output (batch, 512, 1, 1)
            cnn_out_channels = 512
        elif cnn_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.cnn = nn.Sequential(*(list(model.features) + [nn.AdaptiveAvgPool2d(1)]))
            cnn_out_channels = model.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported cnn_name: {cnn_name}")
        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = None  # use BCEWithLogitsLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, 3, H, W)
        batch_size, time_steps, c, h, w = x.shape
        # Flatten batch and time for CNN
        x = x.view(batch_size * time_steps, c, h, w)
        features = self.cnn(x)  # (batch*time, channels, 1, 1)
        features = features.view(batch_size, time_steps, -1)  # (batch, time, channels)
        # GRU
        output, _ = self.gru(features)  # output: (batch, time, hidden)
        # Take last hidden state
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits.squeeze(dim=-1)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for seq, labels in dataloader:
        seq = seq.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * seq.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Returns:
        val_loss, acc, recall_pos, precision_pos, f1_pos, auc
    """
    model.eval()
    running_loss = 0.0

    tp = fp = tn = fn = 0
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for seq, labels in dataloader:
            seq = seq.to(device)
            labels = labels.to(device).float()

            # model returns logits after our imbalance patch
            logits = model(seq)
            loss = criterion(logits, labels)
            running_loss += loss.item() * seq.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            y = labels.int()
            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

            probs_all.append(probs.detach().cpu())
            labels_all.append(labels.detach().cpu())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)) if (precision_pos + recall_pos) > 0 else 0.0

    # AUC needs both classes present
    y_true = torch.cat(labels_all).numpy()
    y_prob = torch.cat(probs_all).numpy()
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    val_loss = running_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    return val_loss, acc, recall_pos, precision_pos, f1_pos, auc

def prepare_splits(data_dir: str, n_splits: int = 5) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Load the labels file and create group K folds.

    Returns lists of training and validation DataFrames for each fold.
    """
    labels_path = os.path.join(data_dir, 'labels.csv')
    labels_df = pd.read_csv(labels_path)
    # Add path column
    labels_df['path'] = labels_df['patient_id'].astype(str).apply(lambda x: os.path.join(data_dir, f"{x}.npy"))
    # Filter rows that exist
    labels_df = labels_df[labels_df['path'].apply(os.path.exists)]
    groups = labels_df['patient_id']
    gkf = GroupKFold(n_splits=n_splits)
    train_folds = []
    val_folds = []
    for train_idx, val_idx in gkf.split(labels_df, labels_df['bowel_injury'], groups):
        train_folds.append(labels_df.iloc[train_idx].copy())
        val_folds.append(labels_df.iloc[val_idx].copy())
    return train_folds, val_folds


def _gaussian_noise(img: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    return torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)


def _train_transforms() -> transforms.Compose:
    """Augmentation pipeline applied per-frame during training."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Lambda(_gaussian_noise),
        transforms.Lambda(lambda img: torch.clamp((img - img.min()) / (img.max() - img.min() + 1e-5), 0.0, 1.0)),
    ])


def _val_transforms() -> transforms.Compose:
    """Deterministic normalisation only — no augmentation."""
    return transforms.Compose([
        transforms.Lambda(lambda img: torch.clamp((img - img.min()) / (img.max() - img.min() + 1e-5), 0.0, 1.0)),
    ])


def _train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    log_writer: csv.DictWriter | None,
) -> dict:
    """Train one cross-validation fold and return best metrics."""
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1}  |  train={len(train_df)}  val={len(val_df)}")
    print("=" * 60)

    train_dataset = BowelInjuryDataset(
        train_df, num_steps=args.num_steps,
        num_slices_per_step=args.num_slices_per_step,
        transform=_train_transforms(),
    )
    val_dataset = BowelInjuryDataset(
        val_df, num_steps=args.num_steps,
        num_slices_per_step=args.num_slices_per_step,
        transform=_val_transforms(),
    )

    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = CNNGRUClassifier(cnn_name=args.cnn_name, hidden_size=args.hidden_size).to(device)

    num_pos = int(train_df["bowel_injury"].sum())
    num_neg = int(len(train_df) - num_pos)
    pw = max(5.0, min(num_neg / max(num_pos, 1), 20.0))
    pos_weight = torch.tensor([pw], device=device)
    print(f"pos_weight={pos_weight.item():.2f}  (neg={num_neg}, pos={num_pos})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    best_f1 = 0.0
    best_metrics: dict = {}
    best_epoch = 0
    epochs_no_improve = 0
    model_save_path = os.path.join(args.data_dir, f"best_bowel_injury_model_fold{fold}.pth")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_recall, val_prec, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Ep {epoch + 1:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={val_acc:.4f}  recall={val_recall:.4f}  "
            f"prec={val_prec:.4f}  f1={val_f1:.4f}  auc={val_auc:.4f}  "
            f"lr={lr_now:.2e}"
        )

        # Append to CSV log
        if log_writer is not None:
            log_writer.writerow({
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_recall": round(val_recall, 6),
                "val_precision": round(val_prec, 6),
                "val_f1": round(val_f1, 6),
                "val_auc": round(val_auc, 6),
                "lr": lr_now,
            })

        scheduler.step(val_f1)

        if val_f1 > best_f1 + args.min_delta:
            best_f1 = val_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_metrics = {
                "fold": fold, "best_epoch": best_epoch,
                "val_acc": val_acc, "val_recall": val_recall,
                "val_precision": val_prec, "val_f1": val_f1, "val_auc": val_auc,
            }
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✓ Saved best model → {model_save_path}  (F1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch + 1}. Best F1={best_f1:.4f} @ epoch {best_epoch}")
                break

    if not best_metrics:
        # No improvement at all; record last epoch metrics
        best_metrics = {
            "fold": fold, "best_epoch": 0,
            "val_acc": val_acc, "val_recall": val_recall,
            "val_precision": val_prec, "val_f1": val_f1, "val_auc": val_auc,
        }

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2.5D bowel injury classifier on processed CT volumes.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing .npy volumes and labels.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs without F1 improvement)")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Min F1 improvement to reset patience")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold folds")
    parser.add_argument("--fold", type=int, default=-1, help="Specific fold index to train (0-based). Default: -1 = all folds")
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-slices-per-step", type=int, default=3)
    parser.add_argument("--cnn-name", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-file", type=str, default="training_log.csv", help="CSV file for per-epoch metrics")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(args.device)
    print(f"Device: {device}")

    train_folds, val_folds = prepare_splits(args.data_dir, n_splits=args.n_splits)

    folds_to_run = list(range(len(train_folds))) if args.fold == -1 else [args.fold]

    log_path = os.path.join(args.data_dir, args.log_file)
    log_fields = ["fold", "epoch", "train_loss", "val_loss", "val_acc", "val_recall", "val_precision", "val_f1", "val_auc", "lr"]

    all_results: list[dict] = []

    with open(log_path, "w", newline="", encoding="utf-8") as log_f:
        log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
        log_writer.writeheader()

        for fold in folds_to_run:
            result = _train_fold(
                fold=fold,
                train_df=train_folds[fold],
                val_df=val_folds[fold],
                args=args,
                device=device,
                log_writer=log_writer,
            )
            all_results.append(result)
            log_f.flush()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    header = f"{'Fold':>5}  {'BestEp':>6}  {'Acc':>7}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'AUC':>7}"
    print(header)
    print("-" * len(header))

    metric_keys = ["val_acc", "val_recall", "val_precision", "val_f1", "val_auc"]
    for r in all_results:
        print(
            f"{r['fold']:>5}  {r['best_epoch']:>6}  "
            f"{r['val_acc']:>7.4f}  {r['val_recall']:>7.4f}  "
            f"{r['val_precision']:>7.4f}  {r['val_f1']:>7.4f}  {r['val_auc']:>7.4f}"
        )

    if len(all_results) > 1:
        print("-" * len(header))
        for key, label in zip(metric_keys, ["Acc", "Recall", "Prec", "F1", "AUC"]):
            vals = [r[key] for r in all_results]
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            print(f"  {label}: {mean:.4f} ± {std:.4f}")

    print(f"\nEpoch-level log saved → {log_path}")


if __name__ == "__main__":
    main()