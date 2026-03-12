from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import pearsonr, spearmanr

from train_scripts.parquet_iterator import WindowConfig, ParquetWindowDataset


@dataclass
class AnalysisConfig:
    input_window: int = 300
    label_window: int = 200
    target_column: str = "close"
    class_boundaries: List[float] | None = None
    exclude_columns: List[str] | None = None
    metrics_lookback: int = 14


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
    high = data["high"]
    low = data["low"]
    close = data["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()


def calculate_volatility(series: pd.Series, period: int) -> pd.Series:
    returns = series.pct_change()
    return returns.rolling(window=period, min_periods=period).std()


def build_feature_vector(input_df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, float]:
    close_series = input_df[config.target_column]

    rsi = calculate_rsi(close_series, config.metrics_lookback)
    ema_fast = calculate_ema(close_series, span=10)
    ema_slow = calculate_ema(close_series, span=20)
    atr = calculate_atr(input_df, period=config.metrics_lookback)
    volatility = calculate_volatility(close_series, config.metrics_lookback)

    features = {
        "rsi_mean": float(rsi.mean()),
        "ema_10_mean": float(ema_fast.mean()),
        "ema_20_mean": float(ema_slow.mean()),
        "atr_mean": float(atr.mean()),
        "volatility_mean": float(volatility.mean()),
        "close_mean": float(close_series.mean()),
        "close_std": float(close_series.std()),
        "volume_mean": float(input_df["volume"].mean()) if "volume" in input_df.columns else np.nan,
    }

    for column in input_df.columns:
        if column in {"open_time", "ticker"}:
            continue
        if pd.api.types.is_numeric_dtype(input_df[column]):
            features[f"{column}_mean"] = float(input_df[column].mean())

    return features


def build_samples(
    parquet_path: str,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    window_config = WindowConfig(
        input_window=config.input_window,
        label_window=config.label_window,
        target_column=config.target_column,
        class_boundaries=config.class_boundaries,
        exclude_columns=config.exclude_columns,
        normalize=False,
    )

    dataset = ParquetWindowDataset(parquet_path, config=window_config)
    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []

    for idx in range(len(dataset)):
        start = idx
        end = idx + config.input_window + config.label_window
        table = ds.dataset(parquet_path, format="parquet").take(list(range(start, end)))
        window_df = table.to_pandas()
        input_df = window_df.iloc[: config.input_window]
        future_df = window_df.iloc[config.input_window :]

        features = build_feature_vector(input_df, config)
        label = dataset._classify_future_window(input_df, future_df)

        feature_rows.append(features)
        labels.append(label)

    feature_table = pd.DataFrame(feature_rows)
    return feature_table, np.array(labels)


def compute_correlations(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    results = []
    label_series = pd.Series(labels, name="label")

    for column in features.columns:
        series = features[column].fillna(0.0)
        pearson_corr, pearson_p = pearsonr(series, label_series)
        spearman_corr, spearman_p = spearmanr(series, label_series)
        results.append(
            {
                "feature": column,
                "pearson_corr": pearson_corr,
                "pearson_p": pearson_p,
                "spearman_corr": spearman_corr,
                "spearman_p": spearman_p,
            }
        )

    return pd.DataFrame(results).sort_values("pearson_corr", ascending=False)


def save_outputs(output_dir: Path, features: pd.DataFrame, labels: np.ndarray, correlations: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    features.assign(label=labels).to_csv(output_dir / "feature_matrix.csv", index=False)
    correlations.to_csv(output_dir / "correlations.csv", index=False)
    stats = features.describe().T
    stats.to_csv(output_dir / "feature_statistics.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Financial feature correlation analysis.")
    parser.add_argument("--parquet", type=str, required=True, help="Path to parquet dataset.")
    parser.add_argument("--output", type=str, default="train_scripts/analysis_output")
    parser.add_argument("--input-window", type=int, default=300)
    parser.add_argument("--label-window", type=int, default=200)
    parser.add_argument("--target-column", type=str, default="close")
    parser.add_argument("--class-boundaries", type=float, nargs="*", default=[-0.002, 0.002])
    parser.add_argument("--exclude-columns", type=str, nargs="*", default=["open_time"])
    parser.add_argument("--metrics-lookback", type=int, default=14)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AnalysisConfig(
        input_window=args.input_window,
        label_window=args.label_window,
        target_column=args.target_column,
        class_boundaries=args.class_boundaries,
        exclude_columns=args.exclude_columns,
        metrics_lookback=args.metrics_lookback,
    )

    features, labels = build_samples(args.parquet, config)
    correlations = compute_correlations(features, labels)
    save_outputs(Path(args.output), features, labels, correlations)


if __name__ == "__main__":
    main()
