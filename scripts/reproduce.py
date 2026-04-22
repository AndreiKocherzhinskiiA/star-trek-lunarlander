from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


DEFAULT_CONFIGS = (
    "configs/baseline/random.yaml",
    "configs/baseline/heuristic.yaml",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate baseline CSV artifacts.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(DEFAULT_CONFIGS),
        help="Config files to run.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for config in args.configs:
        config_path = root / config
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        command = [sys.executable, str(root / "train.py"), "--config", str(config_path)]
        subprocess.run(command, cwd=root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
