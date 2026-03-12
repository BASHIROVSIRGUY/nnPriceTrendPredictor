from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass
class WindowConfig:
    input_window: int = 300
    label_window: int = 200
    target_column: str = "close"
    class_boundaries: Optional[List[float]] = None
    exclude_columns: Optional[List[str]] = None
    normalize: bool = True


class ParquetWindowDataset(Dataset):
    """
    Dataset that loads sliding windows from a parquet file.

    Each sample:
        - input: [input_window, num_features]
        - label: class label derived from the next label_window rows
    """

    def __init__(
        self,
        parquet_path: str,
        config: WindowConfig,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        self.config = config
        self._dataset = ds.dataset(parquet_path, format="parquet")
        self._row_count = pq.ParquetFile(parquet_path).metadata.num_rows
        self._total_window = config.input_window + config.label_window

        if self._row_count < self._total_window:
            raise ValueError(
                f"Недостаточно строк ({self._row_count}) для окна {self._total_window}."
            )

        self.feature_columns = feature_columns or self._infer_feature_columns()
        self._validate_columns()

    def _infer_feature_columns(self) -> List[str]:
        schema = self._dataset.schema
        numeric_types = {
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float",
            "double",
        }
        exclude = set(self.config.exclude_columns or ["open_time"])
        numeric_columns = [
            field.name
            for field in schema
            if field.type.to_string() in numeric_types
        ]
        return [col for col in numeric_columns if col not in exclude]

    def _validate_columns(self) -> None:
        available_columns = set(self._dataset.schema.names)
        missing = [col for col in self.feature_columns if col not in available_columns]
        if missing:
            raise ValueError(f"Отсутствуют колонки в parquet: {missing}")

        if self.config.target_column not in available_columns:
            raise ValueError(
                f"Целевая колонка '{self.config.target_column}' отсутствует в parquet."
            )

    def __len__(self) -> int:
        return self._row_count - self._total_window + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        start = idx
        end = idx + self._total_window
        table = self._dataset.take(list(range(start, end)))
        window_df = table.to_pandas()

        input_df = window_df.iloc[: self.config.input_window]
        future_df = window_df.iloc[self.config.input_window :]

        features = input_df[self.feature_columns].to_numpy(dtype=np.float32)
        if self.config.normalize:
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            features = (features - mean) / std

        label = self._classify_future_window(input_df, future_df)

        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

    def _classify_future_window(self, input_df: pd.DataFrame, future_df: pd.DataFrame) -> int:
        last_close = input_df[self.config.target_column].iloc[-1]
        future_mean = future_df[self.config.target_column].mean()
        if last_close == 0:
            return 0

        pct_change = (future_mean - last_close) / last_close
        boundaries = self.config.class_boundaries or [-0.002, 0.002]
        boundaries = sorted(boundaries)
        return int(np.digitize(pct_change, boundaries))

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    @property
    def num_classes(self) -> int:
        return len(self.config.class_boundaries or [-0.002, 0.002]) + 1
