from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: Path
    episodes: int
    max_steps: int
    seed: int


@dataclass(frozen=True)
class VideoConfig:
    enabled: bool
    record_every: int
    prefix: str


@dataclass(frozen=True)
class EnvConfig:
    id: str
    render_mode: str | None
    continuous: bool
    enable_wind: bool
    wind_power: float
    turbulence_power: float


@dataclass(frozen=True)
class PolicyConfig:
    name: str


@dataclass(frozen=True)
class ProjectConfig:
    run: RunConfig
    env: EnvConfig
    policy: PolicyConfig
    video: VideoConfig

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run"]["output_dir"] = str(self.run.output_dir)
        return data


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = parse_simple_yaml(config_path)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    run = raw.get("run", {})
    env = raw.get("env", {})
    policy = raw.get("policy", {})
    video = raw.get("video", {})

    return ProjectConfig(
        run=RunConfig(
            name=str(run["name"]),
            output_dir=Path(run["output_dir"]),
            episodes=int(run["episodes"]),
            max_steps=int(run["max_steps"]),
            seed=int(run["seed"]),
        ),
        env=EnvConfig(
            id=str(env["id"]),
            render_mode=env.get("render_mode"),
            continuous=bool(env.get("continuous", False)),
            enable_wind=bool(env.get("enable_wind", False)),
            wind_power=float(env.get("wind_power", 15.0)),
            turbulence_power=float(env.get("turbulence_power", 1.5)),
        ),
        policy=PolicyConfig(name=str(policy["name"])),
        video=VideoConfig(
            enabled=bool(video.get("enabled", False)),
            record_every=max(1, int(video.get("record_every", 1))),
            prefix=str(video.get("prefix", "episode")),
        ),
    )


def parse_simple_yaml(path: Path) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = {}
    current_section: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line_without_comment = raw_line.split("#", 1)[0].rstrip()
            if not line_without_comment.strip():
                continue
            indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
            line = line_without_comment.strip()
            if indent == 0:
                if not line.endswith(":"):
                    raise ValueError(f"Invalid section at {path}:{line_number}: {line}")
                current_section = line[:-1]
                data[current_section] = {}
                continue
            if current_section is None:
                raise ValueError(f"Key outside section at {path}:{line_number}: {line}")
            if ":" not in line:
                raise ValueError(f"Invalid key/value at {path}:{line_number}: {line}")
            key, value = line.split(":", 1)
            data[current_section][key.strip()] = parse_scalar(value.strip())
    return data


def parse_scalar(value: str) -> object:
    if value == "" or value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    quoted = (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    )
    if quoted:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
