from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot_baseline import generate_episode_plots, write_benchmark_summary_chart


@dataclass(frozen=True)
class BenchmarkMethod:
    key: str
    label: str
    config: str
    agent_path: str | None = None


DEFAULT_METHODS = (BenchmarkMethod("random", "Random", "configs/baseline/random.yaml"),
    BenchmarkMethod("heuristic", "Heuristic", "configs/baseline/heuristic.yaml"),
    BenchmarkMethod("ppo_early", "Early PPO", "configs/baseline/ppo.yaml", "configs/agent/agent_trained.pt"),
    BenchmarkMethod("ppo_winning", "Winning PPO", "configs/agent/ppo_refine_conservative.yaml", "results/ppo_refine_conservative/agent_best.pt"),)

SUMMARY_FIELDS = ["key", "label", "artifact", "n", "mean_return", "std_return", "ci95_low", "ci95_high", "mean_safe_landing_rate"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the benchmark, plots, and summary tables.")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per evaluation run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="Seeds to evaluate.")
    parser.add_argument("--output-root", default="results/repro_benchmark", help="Directory where regenerated benchmark artifacts will be written.")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_rows = regenerate_benchmark(output_root / "methods", args.seeds, args.episodes)

    summary_csv = output_root / "benchmark_summary.csv"
    summary_md = output_root / "benchmark_summary.md"
    summary_svg = output_root / "benchmark_summary.svg"
    write_summary_csv(benchmark_rows, summary_csv)
    write_summary_markdown(benchmark_rows, summary_md)
    write_benchmark_summary_chart(benchmark_rows, summary_svg)

    print(f"Wrote benchmark CSV to {summary_csv}")
    print(f"Wrote benchmark table to {summary_md}")
    print(f"Wrote benchmark figure to {summary_svg}")
    return 0


def regenerate_benchmark(methods_dir: Path, seeds: list[int], episodes: int) -> list[dict[str, object]]:
    benchmark_rows: list[dict[str, object]] = []
    for method in DEFAULT_METHODS:
        method_dir = methods_dir / method.key
        method_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            seed_dir = method_dir / f"seed_{seed}"
            run_eval(method, seed, episodes, seed_dir)
            generate_episode_plots(seed_dir / "episodes.csv", seed_dir / "plots")
        benchmark_rows.append(aggregate_method(method, method_dir, seeds))
    return benchmark_rows


def run_eval(method: BenchmarkMethod, seed: int, episodes: int, seed_dir: Path) -> None:
    command = [sys.executable, str(ROOT / "eval.py"), "--config", str(ROOT / method.config), "--episodes", str(episodes), "--seed", str(seed), "--output-dir", str(seed_dir)]
    if method.agent_path is not None:
        command.extend(["--agent-path", str(ROOT / method.agent_path)])
    subprocess.run(command, cwd=ROOT, check=True)


def aggregate_method(method: BenchmarkMethod, method_dir: Path, seeds: list[int]) -> dict[str, object]:
    mean_returns: list[float] = []
    safe_landing_rates: list[float] = []
    for seed in seeds:
        row = read_summary_row(method_dir / f"seed_{seed}" / "summary.csv")
        mean_returns.append(float(row["mean_return"]))
        safe_landing_rates.append(float(row.get("reason_safe_landing_rate", 0.0)))

    n = len(mean_returns)
    mean_return = sum(mean_returns) / n
    std_return = statistics.stdev(mean_returns) if n > 1 else 0.0
    ci_half_width = 1.96 * std_return / math.sqrt(n) if n > 1 else 0.0

    return {"key": method.key, "label": method.label,
        "artifact": f"{method.config} + {method.agent_path}" if method.agent_path else method.config,
        "n": n, "mean_return": mean_return, "std_return": std_return,
        "ci95_low": mean_return - ci_half_width, "ci95_high": mean_return + ci_half_width,
        "mean_safe_landing_rate": sum(safe_landing_rates) / n,}

def read_summary_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))

def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def write_summary_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = ["# Benchmark Summary", "", "| Method | Artifact used | Mean return | Std | 95% CI | Mean safe landing rate |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['label']} | `{row['artifact']}` | {float(row['mean_return']):.2f} | {float(row['std_return']):.2f} | "
            f"[{float(row['ci95_low']):.2f}, {float(row['ci95_high']):.2f}] | "
            f"{100.0 * float(row['mean_safe_landing_rate']):.2f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    raise SystemExit(main())
