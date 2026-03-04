"""
Time Series Classifier: TCN + TransformerEncoder
PyTorch 2.x compatible implementation.

Input shape:
    (batch_size=16, seq_len=200, feature_dim=8)

Output:
    (batch_size, 4) — logits for classification
"""

import os
import math
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Synthetic Dataset (Example)
# ============================================================

class TimeSeriesDataset(Dataset):
    """
    Example synthetic dataset for time series classification.

    Each sample:
        x: (seq_len=200, feature_dim=8)
        y: scalar class label in [0, 3]
    """
    TRAIN_CSV_NAME = "train_data.csv"
    VAL_CSV_NAME = "validate_data.csv"

    def __init__(self, dataset_folder_path: str, num_samples: int):
        super().__init__()
        self.folder_path = dataset_folder_path
        self.num_samples = num_samples
        self.seq_len = 200
        self.feature_dim = 8
        self.num_classes = 4

        self.data = self.load_data(dataset_folder_path)
        self.labels = torch.randint(0, self.num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    def load_data(self, folder_path) -> dict[str, torch.Tensor]:
        return {
            "train": pd.read_csv(os.path.join(folder_path, self.TRAIN_CSV_NAME)),
            "val": pd.read_csv(os.path.join(folder_path, self.VAL_CSV_NAME))
        }

    def _train_rows_gen(self):
        with open(os.path.join(self.folder_path, self.TRAIN_CSV_NAME), "r") as csvfile:
            yield csvfile.readlines()

# ============================================================
# 3. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ============================================================
# 4. Temporal Convolutional Block
# ============================================================

class TCNBlock(nn.Module):
    """
    Residual Temporal Convolutional Block.

    Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout
    Residual connection included.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return x + residual


# ============================================================
# 5. Full Model
# ============================================================

class TCNTransformerClassifier(nn.Module):
    """
    Full architecture:
        Input -> TCN -> PositionalEncoding -> TransformerEncoder
              -> Global Average Pooling -> Linear(4)
    """

    def __init__(
        self,
        input_dim: int = 8,
        tcn_channels: int = 64,
        num_tcn_layers: int = 3,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ----- TCN stack -----
        tcn_blocks = []
        in_channels = input_dim

        for i in range(num_tcn_layers):
            dilation = 2 ** i
            tcn_blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=tcn_channels,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = tcn_channels

        self.tcn = nn.Sequential(*tcn_blocks)

        # ----- Transformer -----
        self.pos_encoder = PositionalEncoding(tcn_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tcn_channels,
            nhead=transformer_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # ----- Classifier -----
        self.classifier = nn.Linear(tcn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim=8)
        Returns:
            logits: (batch, num_classes=4)
        """

        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = self.tcn(x)

        # Back to (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Global average pooling over time
        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


# ============================================================
# 6. Training Loop
# ============================================================

def train(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 5,
) -> None:

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(epochs):
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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")


# ============================================================
# 7. Main Execution
# ============================================================

if __name__ == "__main__":

    set_seed(42)

    dataset_path = "/home/dyadya/PycharmProjects/trade/nnPriceTrendPredictor/Data/_dataset_csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TimeSeriesDataset(dataset_path, num_samples=1000)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = TCNTransformerClassifier()

    # Sanity check: forward pass
    sample_input = torch.randn(16, 200, 8).to(device)
    model.to(device)
    output = model(sample_input)
    print("Output shape:", output.shape)  # should be (16, 4)

    train(model, dataloader, device, epochs=5)
