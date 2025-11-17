"""Quick visualization helpers for prediction parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib optional
    plt = None
CURR_DIR = Path(__file__).resolve().parent
PLOT_DIR = CURR_DIR / "runs" / "plots"
CSV_DIR = CURR_DIR / "runs" / "csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect preds_test.parquet quickly.")
    parser.add_argument("--preds", required=True, help="Path to preds_test.parquet or csv.")
    parser.add_argument(
        "--limit", type=int, default=10000, help="Number of rows to sample for plotting (default: 10k)."
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional path to export a CSV version of the predictions.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=PLOT_DIR,
        help="If provided, saves a scatter plot (label vs prediction) to this PNG path.",
    )
    return parser.parse_args()


def load_predictions(path: str | Path, limit: int) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=42).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    df = load_predictions(args.preds, args.limit)
    print("Preview:")
    print(df.head(20))
    print("\nDescribe:")
    print(df.describe())

    if args.export_csv:
        csv_path = Path(args.export_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Exported CSV to {csv_path}")

    if args.plot_path:
        if plt is None:
            raise RuntimeError("matplotlib is required for --plot-path")
        if {"prediction", "rating"} <= set(df.columns):
            plt.figure(figsize=(6, 6))
            plt.scatter(df["rating"], df["prediction"], s=6, alpha=0.4)
            plt.xlabel("Label")
            plt.ylabel("Prediction")
            plt.title("Prediction vs Label")
        else:
            plt.figure(figsize=(6, 4))
            df["prediction"].hist(bins=50)
            plt.xlabel("Prediction")
            plt.ylabel("Count")
            plt.title("Prediction distribution")
        plot_path = Path(args.plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
