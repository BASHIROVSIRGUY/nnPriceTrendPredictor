from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    data = pd.read_csv(csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(parquet_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet format.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output Parquet file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    convert_csv_to_parquet(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
