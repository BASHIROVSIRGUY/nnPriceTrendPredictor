"""
Time Series Classifier: TCN + TransformerEncoder

Input:
    300 rows -> model input
Target:
    200 rows -> label computed by classifier
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset

from train_scripts.parquet_iterator import ParquetWindowDataset, WindowConfig


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ============================================================
# 3. TCN Block
# ============================================================

class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))

        if self.downsample is not None:
            residual = self.downsample(residual)

        return x + residual


# ============================================================
# 4. Full Model
# ============================================================

@dataclass
class ModelConfig:
    input_dim: int
    num_classes: int
    tcn_channels: List[int]
    kernel_size: int
    transformer_heads: int
    transformer_layers: int
    transformer_ff_dim: int
    dropout: float
    classifier_hidden: List[int]


class TCNTransformerClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        tcn_blocks = []
        in_channels = config.input_dim
        for idx, out_channels in enumerate(config.tcn_channels):
            tcn_blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    dilation=2 ** idx,
                    dropout=config.dropout,
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*tcn_blocks)
        self.pos_encoder = PositionalEncoding(in_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )

        classifier_layers: List[nn.Module] = []
        classifier_in = in_channels
        for hidden_dim in config.classifier_hidden:
            classifier_layers.append(nn.Linear(classifier_in, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(config.dropout))
            classifier_in = hidden_dim
        classifier_layers.append(nn.Linear(classifier_in, config.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================================
# 5. Training / Evaluation
# ============================================================

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    output_dir: Path


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

    return logits, probabilities, targets


def save_metrics(
    output_dir: Path,
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    predictions = probabilities.argmax(axis=1)
    num_classes = probabilities.shape[1]

    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision_macro": precision_score(targets, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
        "f2_macro": fbeta_score(targets, predictions, beta=2.0, average="macro", zero_division=0),
    }

    try:
        targets_binarized = label_binarize(targets, classes=list(range(num_classes)))
        metrics["roc_auc_ovr"] = roc_auc_score(
            targets_binarized,
            probabilities,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        metrics["roc_auc_ovr"] = float("nan")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    for class_idx in range(num_classes):
        if (targets == class_idx).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(
            (targets == class_idx).astype(int),
            probabilities[:, class_idx],
        )
        plt.plot(fpr, tpr, label=f"Class {class_idx}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png")
    plt.close()


def split_dataset(dataset: torch.utils.data.Dataset, train_ratio: float) -> Tuple[Subset, Subset]:
    train_size = int(len(dataset) * train_ratio)
    indices = np.arange(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ============================================================
# 6. Main
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train time series classifier on parquet data.")
    parser.add_argument("--parquet", type=str, required=True, help="Path to parquet dataset.")
    parser.add_argument("--output", type=str, default="train_scripts/output", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input-window", type=int, default=300)
    parser.add_argument("--label-window", type=int, default=200)
    parser.add_argument("--target-column", type=str, default="close")
    parser.add_argument("--class-boundaries", type=float, nargs="*", default=[-0.002, 0.002])
    parser.add_argument("--exclude-columns", type=str, nargs="*", default=["open_time"])

    parser.add_argument("--tcn-channels", type=int, nargs="*", default=[64, 64, 64])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier-hidden", type=int, nargs="*", default=[64])

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_config = WindowConfig(
        input_window=args.input_window,
        label_window=args.label_window,
        target_column=args.target_column,
        class_boundaries=args.class_boundaries,
        exclude_columns=args.exclude_columns,
    )

    dataset = ParquetWindowDataset(args.parquet, config=window_config)
    train_dataset, val_dataset = split_dataset(dataset, args.train_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_config = ModelConfig(
        input_dim=dataset.num_features,
        num_classes=dataset.num_classes,
        tcn_channels=args.tcn_channels,
        kernel_size=args.kernel_size,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        dropout=args.dropout,
        classifier_hidden=args.classifier_hidden,
    )

    model = TCNTransformerClassifier(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        logits, probabilities, targets = evaluate(model, val_loader, device)
        preds = probabilities.argmax(axis=1)
        val_accuracy = accuracy_score(targets, preds)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

    save_metrics(output_dir, targets, probabilities)
    torch.save(model.state_dict(), output_dir / "model.pth")


if __name__ == "__main__":
    main()
