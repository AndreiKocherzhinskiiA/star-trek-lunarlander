from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from start_trek.baseline import run_policy
from start_trek.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Start Trek training/baseline job.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", help="Override config output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    records = run_policy(config, output_dir=output_dir)
    mean_return = sum(record.return_ for record in records) / len(records)
    print(f"Completed {len(records)} episodes. Mean return: {mean_return:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
