import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT_DIR = Path("data_analysis_scripts/output")


def load_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if "open_time" in data.columns:
        data["open_time"] = pd.to_datetime(data["open_time"], unit="ms", errors="coerce")
    return data


def get_numeric_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.select_dtypes(include=["number"]).copy()


def save_basic_statistics(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data.describe().to_csv(output_dir / "basic_statistics.csv")
    data.skew(numeric_only=True).to_frame("skew").to_csv(output_dir / "skewness.csv")
    data.kurt(numeric_only=True).to_frame("kurtosis").to_csv(output_dir / "kurtosis.csv")
    data.isna().sum().to_frame("missing_values").to_csv(output_dir / "missing_values.csv")


def save_correlation_artifacts(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    correlation_matrix = data.corr()
    correlation_matrix.to_csv(output_dir / "correlation_matrix.csv")

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    return correlation_matrix


def save_distribution_plots(data: pd.DataFrame, output_dir: Path) -> None:
    for column in data.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[column].dropna(), kde=True)
        plt.title(f"Distribution: {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"distribution_{column}.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=data[column].dropna())
        plt.title(f"Boxplot: {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{column}.png")
        plt.close()


def save_time_series_plots(data: pd.DataFrame, output_dir: Path) -> None:
    if "open_time" not in data.columns:
        return

    time_series_columns = [col for col in ["open", "high", "low", "close", "volume"] if col in data.columns]
    for column in time_series_columns:
        plt.figure(figsize=(10, 4))
        sns.lineplot(x=data["open_time"], y=data[column])
        plt.title(f"Time Series: {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"timeseries_{column}.png")
        plt.close()


def run_factor_analysis(data: pd.DataFrame, output_dir: Path, n_factors: int) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    factor_analysis = FactorAnalysis(n_components=n_factors, random_state=42)
    factor_scores = factor_analysis.fit_transform(scaled_data)
    loadings = pd.DataFrame(
        factor_analysis.components_.T,
        index=data.columns,
        columns=[f"factor_{idx + 1}" for idx in range(n_factors)],
    )

    loadings.to_csv(output_dir / "factor_loadings.csv")

    factor_scores_df = pd.DataFrame(
        factor_scores,
        columns=[f"factor_{idx + 1}" for idx in range(n_factors)],
    )
    factor_scores_df.to_csv(output_dir / "factor_scores.csv", index=False)

    return loadings


def select_top_features(loadings: pd.DataFrame, top_k: int) -> list[str]:
    feature_scores = loadings.abs().sum(axis=1)
    top_features = feature_scores.sort_values(ascending=False).head(top_k).index.tolist()
    return top_features


def run_clustering(data: pd.DataFrame, output_dir: Path, features: list[str], n_clusters: int) -> None:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    clustered_data = data.copy()
    clustered_data["cluster"] = cluster_labels
    clustered_data.to_csv(output_dir / "clustered_data.csv", index=False)

    if len(features) == 1:
        plt.figure(figsize=(8, 4))
        sns.scatterplot(x=clustered_data.index, y=clustered_data[features[0]], hue=clustered_data["cluster"], palette="tab10")
        plt.title("Clusters by Selected Feature")
    elif len(features) == 2:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=clustered_data[features[0]], y=clustered_data[features[1]], hue=clustered_data["cluster"], palette="tab10")
        plt.title("Clusters (2D)")
    else:
        pca = PCA(n_components=2, random_state=42)
        projected = pca.fit_transform(scaled_features)
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=clustered_data["cluster"], palette="tab10")
        plt.title("Clusters (PCA Projection)")

    plt.tight_layout()
    plt.savefig(output_dir / "clusters.png")
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze dataset, correlations, factor analysis, and clustering.")
    parser.add_argument(
        "--input",
        type=str,
        default="data_analysis_scripts/example_data.csv",
        help="Path to input CSV data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for reports and plots.",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=3,
        help="Number of factors for factor analysis.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of clusters for KMeans.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=3,
        help="Number of most significant features to use in clustering.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path)
    numeric_data = get_numeric_data(data)

    save_basic_statistics(numeric_data, output_dir)
    save_correlation_artifacts(numeric_data, output_dir)
    save_distribution_plots(numeric_data, output_dir)
    save_time_series_plots(data, output_dir)

    n_factors = max(1, min(args.factors, numeric_data.shape[1]))
    factor_loadings = run_factor_analysis(numeric_data, output_dir, n_factors=n_factors)

    top_features_count = max(1, min(args.top_features, numeric_data.shape[1]))
    top_features = select_top_features(factor_loadings, top_features_count)

    run_clustering(numeric_data, output_dir, features=top_features, n_clusters=args.clusters)


if __name__ == "__main__":
    main()
