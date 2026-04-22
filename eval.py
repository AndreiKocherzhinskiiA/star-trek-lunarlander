from __future__ import annotations
import argparse
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from start_trek.baseline import run_policy
from start_trek.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Start Trek policy config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes.")
    parser.add_argument("--output-dir", help="Override eval output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_config = replace(config.run, episodes=args.episodes)
    eval_config = replace(config, run=run_config)
    output_dir = Path(args.output_dir) if args.output_dir else config.run.output_dir / "eval"

    records = run_policy(eval_config, output_dir=output_dir)
    mean_return = sum(record.return_ for record in records) / len(records)
    print(f"Evaluated {len(records)} episodes. Mean return: {mean_return:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
