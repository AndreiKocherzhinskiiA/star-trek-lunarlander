from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from start_trek.agent import PPOAgent
from start_trek.baseline import (
    build_policy,
    classify_termination,
    make_env,
    run_policy,
)
from start_trek.config import ProjectConfig, load_config


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    advantages: list[float] = []
    gae = 0.0
    values_ext = values + [last_value]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)

    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def write_training_metrics(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_training_summary(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def train_ppo(
    config: ProjectConfig,
    env,
    policy: PPOAgent,
    target_dir: Path,
    total_timesteps: int,
) -> Path:
    ppo_config = config.ppo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.device = torch.device(device)
    policy.model.to(policy.device)
    for group in policy.optimizer.param_groups:
        group["lr"] = ppo_config.learning_rate

    metrics_rows: list[dict[str, object]] = []
    completed_episode_returns: list[float] = []
    completed_episode_lengths: list[int] = []
    completed_episode_reasons: list[str] = []

    total_steps = 0
    update = 0
    episode_index = 0
    episode_seed = config.run.seed
    episode_return = 0.0
    episode_length = 0

    observation, _ = env.reset(seed=episode_seed)
    env.action_space.seed(episode_seed)

    while total_steps < total_timesteps:
        obs_buf: list[np.ndarray] = []
        actions_buf = []
        logp_buf = []
        rewards_buf: list[float] = []
        values_buf = []
        dones_buf: list[bool] = []

        rollout_completed_returns: list[float] = []
        rollout_completed_lengths: list[int] = []
        rollout_completed_reasons: list[str] = []

        steps_this_rollout = min(ppo_config.steps_per_rollout, total_timesteps - total_steps)
        for _ in range(steps_this_rollout):
            action_t, logp_t, value_t = policy.sample_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(int(action_t.item()))

            obs_buf.append(np.asarray(observation, dtype=np.float32))
            actions_buf.append(action_t.detach().cpu())
            logp_buf.append(logp_t.detach().cpu())
            values_buf.append(value_t.detach().cpu())
            rewards_buf.append(float(reward))
            dones_buf.append(bool(terminated or truncated))

            episode_return += float(reward)
            episode_length += 1
            total_steps += 1

            if terminated or truncated:
                reason = classify_termination(
                    next_observation,
                    episode_return,
                    terminated,
                    truncated,
                )
                rollout_completed_returns.append(episode_return)
                rollout_completed_lengths.append(episode_length)
                rollout_completed_reasons.append(reason)
                completed_episode_returns.append(episode_return)
                completed_episode_lengths.append(episode_length)
                completed_episode_reasons.append(reason)

                episode_index += 1
                episode_seed = config.run.seed + episode_index
                observation, _ = env.reset(seed=episode_seed)
                env.action_space.seed(episode_seed)
                episode_return = 0.0
                episode_length = 0
            else:
                observation = next_observation

        last_value = 0.0
        if rewards_buf and not dones_buf[-1]:
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=policy.device,
                ).unsqueeze(0)
                _, last_value_t = policy.model(obs_t)
                last_value = float(last_value_t.squeeze(0).item())

        advantages, returns = compute_gae(
            rewards_buf,
            [float(value.item()) for value in values_buf],
            dones_buf,
            last_value,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        )

        obs_tensor = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=policy.device)
        actions_tensor = torch.stack(actions_buf).to(device=policy.device)
        old_logp_tensor = torch.stack(logp_buf).to(device=policy.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=policy.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=policy.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std(unbiased=False) + 1e-8
        )

        update_metrics = policy.update(
            obs_tensor,
            actions_tensor,
            old_logp_tensor,
            returns_tensor,
            advantages_tensor,
            epochs=ppo_config.update_epochs,
            minibatch_size=ppo_config.minibatch_size,
            clip_epsilon=ppo_config.clip_epsilon,
            value_coef=ppo_config.value_coef,
            entropy_coef=ppo_config.entropy_coef,
            max_grad_norm=ppo_config.max_grad_norm,
        )

        update += 1
        safe_landing_count = sum(reason == "safe_landing" for reason in rollout_completed_reasons)
        timeout_count = sum(reason == "timeout" for reason in rollout_completed_reasons)
        mean_episode_return = (
            sum(rollout_completed_returns) / len(rollout_completed_returns)
            if rollout_completed_returns
            else None
        )
        mean_episode_length = (
            sum(rollout_completed_lengths) / len(rollout_completed_lengths)
            if rollout_completed_lengths
            else None
        )
        safe_landing_rate = (
            safe_landing_count / len(rollout_completed_reasons)
            if rollout_completed_reasons
            else None
        )

        metrics_rows.append(
            {
                "update": update,
                "total_steps": total_steps,
                "episodes_completed": len(completed_episode_returns),
                "rollout_steps": steps_this_rollout,
                "rollout_episodes_completed": len(rollout_completed_returns),
                "mean_episode_return": mean_episode_return,
                "mean_episode_length": mean_episode_length,
                "safe_landing_rate": safe_landing_rate,
                "timeout_count": timeout_count,
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
                "approx_kl": update_metrics["approx_kl"],
            }
        )

        if update % ppo_config.log_every_updates == 0:
            completed_text = len(rollout_completed_returns)
            mean_return_text = (
                f"{mean_episode_return:.2f}" if mean_episode_return is not None else "n/a"
            )
            safe_rate_text = (
                f"{safe_landing_rate:.2%}" if safe_landing_rate is not None else "n/a"
            )
            print(
                " | ".join(
                    [
                        f"update={update}",
                        f"steps={total_steps}",
                        f"episodes={completed_text}",
                        f"mean_return={mean_return_text}",
                        f"safe_landing_rate={safe_rate_text}",
                        f"policy_loss={update_metrics['policy_loss']:.4f}",
                        f"value_loss={update_metrics['value_loss']:.4f}",
                    ]
                )
            )

    checkpoint_dir = Path(target_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "agent_trained.pt"
    policy.save(checkpoint_path)
    write_training_metrics(metrics_rows, checkpoint_dir / "training_metrics.csv")

    summary = {
        "total_timesteps": total_steps,
        "updates": update,
        "completed_episodes": len(completed_episode_returns),
        "mean_episode_return": (
            sum(completed_episode_returns) / len(completed_episode_returns)
            if completed_episode_returns
            else None
        ),
        "mean_episode_length": (
            sum(completed_episode_lengths) / len(completed_episode_lengths)
            if completed_episode_lengths
            else None
        ),
        "safe_landing_rate": (
            sum(reason == "safe_landing" for reason in completed_episode_reasons)
            / len(completed_episode_reasons)
            if completed_episode_reasons
            else None
        ),
        "timeout_rate": (
            sum(reason == "timeout" for reason in completed_episode_reasons)
            / len(completed_episode_reasons)
            if completed_episode_reasons
            else None
        ),
    }
    write_training_summary(summary, checkpoint_dir / "training_summary.json")
    print(f"Saved trained agent to {checkpoint_path}")
    return checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Start Trek training/baseline job.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", help="Override config output directory.")
    parser.add_argument("--agent-path", help="Path to saved agent checkpoint (for PPO).")
    parser.add_argument("--train", action="store_true", help="Run training loop instead of evaluation.")
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Override total training timesteps from config when --train is used.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    target_dir = output_dir or config.run.output_dir

    env_config = config
    if args.train and config.video.enabled:
        env_config = replace(config, video=replace(config.video, enabled=False))

    env = make_env(env_config, output_dir=target_dir)
    policy = build_policy(
        config.policy.name,
        env.observation_space,
        env.action_space,
        agent_path=args.agent_path,
    )

    if args.train:
        if not isinstance(policy, PPOAgent):
            obs_shape = cast(tuple[int, ...] | None, getattr(env.observation_space, "shape", None))
            if not obs_shape:
                raise ValueError("PPO requires an observation space with a shape.")
            obs_dim = int(obs_shape[0])
            action_dim = int(cast(Any, env.action_space).n)
            policy = PPOAgent(
                obs_dim,
                action_dim,
                lr=config.ppo.learning_rate,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        total_timesteps = args.timesteps or config.ppo.total_timesteps
        try:
            train_ppo(config, env, policy, target_dir, total_timesteps)
        finally:
            env.close()
        return 0

    env.close()
    records = run_policy(config, output_dir=target_dir, policy=policy)

    if isinstance(policy, PPOAgent):
        checkpoint_dir = Path(target_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "agent.pt"
        policy.save(checkpoint_path)
        print(f"Saved agent to {checkpoint_path}")

    mean_return = sum(record.return_ for record in records) / len(records)
    print(f"Completed {len(records)} episodes. Mean return: {mean_return:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
