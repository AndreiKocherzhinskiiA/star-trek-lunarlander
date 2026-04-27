from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot baseline episode metrics from a CSV file.")
    parser.add_argument("episodes_csv", help="Path to episodes.csv")
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where plots will be written.",
    )
    args = parser.parse_args()

    episodes_path = Path(args.episodes_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(episodes_path)
    if df.empty:
        raise ValueError("episodes.csv is empty")

    returns_path = output_dir / "returns.png"
    lengths_path = output_dir / "lengths.png"
    reasons_path = output_dir / "termination_reasons.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["episode"], df["return"], label="return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode returns")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(returns_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["episode"], df["length"], color="orange", label="length")
    plt.xlabel("Episode")
    plt.ylabel("Episode length")
    plt.title("Episode lengths")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(lengths_path, dpi=150)
    plt.close()

    reason_counts = df["termination_reason"].value_counts().sort_index()
    plt.figure(figsize=(8, 4.5))
    reason_counts.plot(kind="bar", color="seagreen")
    plt.xlabel("Termination reason")
    plt.ylabel("Count")
    plt.title("Termination reasons")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(reasons_path, dpi=150)
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
