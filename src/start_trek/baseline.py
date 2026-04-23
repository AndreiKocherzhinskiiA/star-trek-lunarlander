from __future__ import annotations
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from start_trek.config import ProjectConfig


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    seed: int
    return_: float
    length: int
    terminated: bool
    truncated: bool
    termination_reason: str

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["return"] = row.pop("return_")
        return row


class Policy(Protocol):
    def act(self, observation: np.ndarray) -> int:
        ...


class RandomPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def act(self, observation: np.ndarray) -> int:
        del observation
        return int(self.action_space.sample())


class HeuristicPolicy:
    """Small discrete controller adapted to LunarLander's 8-value state."""

    def act(self, observation: np.ndarray) -> int:
        x, _, _, vy, angle, angular_velocity, left_contact, right_contact = observation

        if left_contact and right_contact:
            return 0

        angle_target = x * 0.5
        angle_error = angle - angle_target
        hover_target = 0.55 * abs(x)
        descent_error = vy + hover_target

        if angle_error > 0.08 or angular_velocity > 0.12:
            return 3
        if angle_error < -0.08 or angular_velocity < -0.12:
            return 1
        if descent_error < -0.12:
            return 2
        return 0


def make_env(config: ProjectConfig, output_dir: Path | None = None) -> gym.Env:
    render_mode = config.env.render_mode
    if config.video.enabled and render_mode != "rgb_array":
        render_mode = "rgb_array"

    kwargs: dict[str, Any] = {
        "render_mode": render_mode,
        "continuous": config.env.continuous,
        "enable_wind": config.env.enable_wind,
        "wind_power": config.env.wind_power,
        "turbulence_power": config.env.turbulence_power,
    }
    env = gym.make(config.env.id, **kwargs)

    if config.video.enabled:
        base_output_dir = output_dir or config.run.output_dir
        video_dir = base_output_dir / "videos"
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode: episode % config.video.record_every == 0,
            name_prefix=config.video.prefix,
            disable_logger=True,
        )

    return env


def build_policy(name: str, action_space) -> Policy:
    normalized = name.lower().strip()
    if normalized == "random":
        return RandomPolicy(action_space)
    if normalized == "heuristic":
        return HeuristicPolicy()
    raise ValueError(f"Unknown policy: {name}")


def classify_termination(
    final_observation: np.ndarray,
    total_return: float,
    terminated: bool,
    truncated: bool,
) -> str:
    if truncated:
        return "timeout"
    if not terminated:
        return "unknown"

    x_position = float(final_observation[0])
    left_contact = bool(final_observation[6])
    right_contact = bool(final_observation[7])

    if abs(x_position) >= 1.0:
        return "out_of_view"
    if total_return >= 100:
        return "safe_landing"
    if total_return <= -100:
        return "crash"
    if left_contact and right_contact:
        return "sleep"
    return "crash"


def run_policy(config: ProjectConfig, output_dir: Path | None = None) -> list[EpisodeRecord]:
    target_dir = output_dir or config.run.output_dir
    env = make_env(config, output_dir=target_dir)
    policy = build_policy(config.policy.name, env.action_space)
    records: list[EpisodeRecord] = []

    try:
        for episode in range(config.run.episodes):
            episode_seed = config.run.seed + episode
            observation, _ = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)

            total_return = 0.0
            terminated = False
            truncated = False
            step_count = 0

            for step_count in range(1, config.run.max_steps + 1):
                action = policy.act(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                if terminated or truncated:
                    break

            records.append(
                EpisodeRecord(
                    episode=episode,
                    seed=episode_seed,
                    return_=total_return,
                    length=step_count,
                    terminated=terminated,
                    truncated=truncated,
                    termination_reason=classify_termination(
                        observation, total_return, terminated, truncated
                    ),
                )
            )
    finally:
        env.close()

    write_outputs(config, records, target_dir)
    return records


def write_outputs(
    config: ProjectConfig, records: list[EpisodeRecord], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, output_dir / "config.json")
    write_episode_csv(records, output_dir / "episodes.csv")
    write_summary_csv(records, output_dir / "summary.csv")


def write_config(config: ProjectConfig, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_json_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_episode_csv(records: list[EpisodeRecord], path: Path) -> None:
    fieldnames = [
        "episode",
        "seed",
        "return",
        "length",
        "terminated",
        "truncated",
        "termination_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)


def write_summary_csv(records: list[EpisodeRecord], path: Path) -> None:
    if not records:
        raise ValueError("Cannot summarize an empty run.")

    returns = [record.return_ for record in records]
    lengths = [record.length for record in records]
    reasons = sorted({record.termination_reason for record in records})

    row: dict[str, object] = {
        "episodes": len(records),
        "mean_return": sum(returns) / len(returns),
        "best_return": max(returns),
        "worst_return": min(returns),
        "mean_length": sum(lengths) / len(lengths),
        "terminated_count": sum(record.terminated for record in records),
        "truncated_count": sum(record.truncated for record in records),
    }
    for reason in reasons:
        count = sum(
            record.termination_reason == reason for record in records
        )
        row[f"reason_{reason}_count"] = count
        row[f"reason_{reason}_rate"] = count / len(records)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
