from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

WIDTH = 900
HEIGHT = 520
MARGIN_LEFT = 72
MARGIN_RIGHT = 28
MARGIN_TOP = 48
MARGIN_BOTTOM = 64
SUMMARY_COLORS = ("#2563eb", "#d97706", "#7c3aed", "#15803d")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot baseline episode metrics from a CSV file.")
    parser.add_argument("episodes_csv", help="Path to episodes.csv")
    parser.add_argument("--output-dir", default="plots", help="Directory where SVG plots will be written.")
    args = parser.parse_args()
    generate_episode_plots(Path(args.episodes_csv), Path(args.output_dir))
    return 0

def generate_episode_plots(episodes_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(episodes_path)
    if not rows:
        raise ValueError("episodes.csv is empty")
    line_chart(rows, value_key="return", title="Episode returns", y_label="Return", color="#2563eb", path=output_dir / "returns.svg")
    line_chart(rows, value_key="length", title="Episode lengths", y_label="Episode length", color="#d97706", path=output_dir / "lengths.svg")
    bar_chart(Counter(row["termination_reason"] for row in rows), title="Termination reasons", y_label="Count", color="#15803d", path=output_dir / "termination_reasons.svg")

def write_benchmark_summary_chart(rows: list[dict[str, object]], path: Path) -> None:
    means = [float(row["mean_return"]) for row in rows]
    lows = [float(row["ci95_low"]) for row in rows]
    highs = [float(row["ci95_high"]) for row in rows]
    y_min = min(lows + [0.0])
    y_max = max(highs + [0.0])
    y_low, y_high = padded_range(y_min, y_max)
    zero_y = scale(0.0, y_low, y_high, HEIGHT - MARGIN_BOTTOM, MARGIN_TOP)
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    slot_width = plot_width / max(1, len(rows))
    bar_width = slot_width * 0.5

    body = [axes("Benchmark Mean Return by Method", "Method", "Mean return", y_low, y_high)]
    body.append(f'<line x1="{MARGIN_LEFT}" y1="{zero_y:.2f}" x2="{WIDTH - MARGIN_RIGHT}" '
        f'y2="{zero_y:.2f}" stroke="#9ca3af" stroke-dasharray="4 4" />')
    for index, row in enumerate(rows):
        mean_value = float(row["mean_return"])
        low_value = float(row["ci95_low"])
        high_value = float(row["ci95_high"])
        x = MARGIN_LEFT + index * slot_width + (slot_width - bar_width) / 2
        y = scale(mean_value, y_low, y_high, HEIGHT - MARGIN_BOTTOM, MARGIN_TOP)
        bar_top = min(y, zero_y)
        bar_height = abs(zero_y - y)
        center_x = x + bar_width / 2
        ci_top = scale(high_value, y_low, y_high, HEIGHT - MARGIN_BOTTOM, MARGIN_TOP)
        ci_bottom = scale(low_value, y_low, y_high, HEIGHT - MARGIN_BOTTOM, MARGIN_TOP)
        color = SUMMARY_COLORS[index % len(SUMMARY_COLORS)]

        body.append(f'<rect x="{x:.2f}" y="{bar_top:.2f}" width="{bar_width:.2f}" '
            f'height="{max(bar_height, 1.0):.2f}" fill="{color}" />')
        body.append(f'<line x1="{center_x:.2f}" y1="{ci_top:.2f}" x2="{center_x:.2f}" '
            f'y2="{ci_bottom:.2f}" stroke="#111827" stroke-width="2" />')
        body.append(f'<line x1="{center_x - 10:.2f}" y1="{ci_top:.2f}" x2="{center_x + 10:.2f}" '
            f'y2="{ci_top:.2f}" stroke="#111827" stroke-width="2" />')
        body.append(f'<line x1="{center_x - 10:.2f}" y1="{ci_bottom:.2f}" x2="{center_x + 10:.2f}" '
            f'y2="{ci_bottom:.2f}" stroke="#111827" stroke-width="2" />')
        body.append(f'<text x="{center_x:.2f}" y="{HEIGHT - 54}" text-anchor="middle" font-size="12">{escape(str(row["label"]))}</text>')
        body.append(f'<text x="{center_x:.2f}" y="{bar_top - 10:.2f}" text-anchor="middle" font-size="12">{mean_value:.2f}</text>')

    write_svg(path, body)

def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append({"episode": int(row["episode"]), "return": float(row["return"]), "length": int(row["length"]), "termination_reason": row["termination_reason"]})
    return rows


def line_chart(rows: list[dict[str, object]], value_key: str, title: str, y_label: str, color: str, path: Path) -> None:
    x_values = [int(row["episode"]) for row in rows]
    y_values = [float(row[value_key]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = padded_range(min(y_values), max(y_values))
    points = [f"{scale(x, x_min, x_max, MARGIN_LEFT, WIDTH - MARGIN_RIGHT):.2f},"
        f"{scale(y, y_min, y_max, HEIGHT - MARGIN_BOTTOM, MARGIN_TOP):.2f}" for x, y in zip(x_values, y_values)]
    body = [axes(title, "Episode", y_label, y_min, y_max), f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}" />']
    write_svg(path, body)


def bar_chart(counts: Counter[str], title: str, y_label: str, color: str, path: Path) -> None:
    labels = sorted(counts)
    max_count = max(counts.values())
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    slot_width = plot_width / max(1, len(labels))
    bar_width = slot_width * 0.6
    zero_y = HEIGHT - MARGIN_BOTTOM

    body = [axes(title, "Termination reason", y_label, 0, float(max_count))]
    for index, label in enumerate(labels):
        count = counts[label]
        x = MARGIN_LEFT + index * slot_width + (slot_width - bar_width) / 2
        y = scale(count, 0, max_count, zero_y, MARGIN_TOP)
        height = zero_y - y
        text_x = x + bar_width / 2
        body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height:.2f}" fill="{color}" />')
        body.append(f'<text x="{text_x:.2f}" y="{HEIGHT - 30}" text-anchor="middle" font-size="12">{escape(label)}</text>')
        body.append(f'<text x="{text_x:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12">{count}</text>')
    write_svg(path, body)


def axes(title: str, x_label: str, y_label: str, y_min: float, y_max: float) -> str:
    zero_y = HEIGHT - MARGIN_BOTTOM
    right_x = WIDTH - MARGIN_RIGHT
    tick_parts = []
    for index in range(5):
        ratio = index / 4
        y = MARGIN_TOP + ratio * (zero_y - MARGIN_TOP)
        value = y_max - ratio * (y_max - y_min)
        tick_parts.append(f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{right_x}" y2="{y:.2f}" stroke="#e5e7eb" />')
        tick_parts.append(f'<text x="{MARGIN_LEFT - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:.1f}</text>')
    return "\n".join([f'<text x="{WIDTH / 2}" y="28" text-anchor="middle" font-size="20" font-weight="700">{escape(title)}</text>',
        *tick_parts,
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{zero_y}" stroke="#111827" />',
        f'<line x1="{MARGIN_LEFT}" y1="{zero_y}" x2="{right_x}" y2="{zero_y}" stroke="#111827" />',
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 12}" text-anchor="middle" font-size="14">{escape(x_label)}</text>',
        f'<text x="18" y="{HEIGHT / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {HEIGHT / 2})">{escape(y_label)}</text>',])

def padded_range(low: float, high: float) -> tuple[float, float]:
    if low == high:
        padding = max(1.0, abs(low) * 0.1)
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return low - padding, high + padding


def scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if low == high:
        return (out_low + out_high) / 2
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def write_svg(path: Path, body: list[str]) -> None:
    svg = "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', '<rect width="100%" height="100%" fill="white" />', '<g font-family="Arial, sans-serif" fill="#111827">', *body, "</g>", "</svg>", ""])
    path.write_text(svg, encoding="utf-8")

if __name__ == "__main__":
    raise SystemExit(main())
