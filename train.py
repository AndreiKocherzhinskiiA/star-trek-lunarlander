from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any, cast
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from start_trek.baseline import run_policy, make_env, build_policy
from start_trek.config import load_config
from start_trek.agent import PPOAgent
import torch
import numpy as np


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """Compute generalized advantage estimates and returns for one rollout."""
    advantages: list[float] = []
    gae = 0.0
    values_ext = values + [last_value]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Start Trek training/baseline job.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", help="Override config output directory.")
    parser.add_argument("--agent-path", help="Path to saved agent checkpoint (for PPO).")
    parser.add_argument("--train", action="store_true", help="Run training loop instead of evaluation.")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total training timesteps when --train is used.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    target_dir = output_dir or config.run.output_dir


    env = make_env(config, output_dir=target_dir)
    policy = build_policy(
        config.policy.name,
        env.observation_space,
        env.action_space,
        agent_path=args.agent_path,
    )

    if args.train:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(policy, PPOAgent):
            agent = policy
        else:
            obs_shape = cast(tuple[int, ...] | None, getattr(env.observation_space, "shape", None))
            if not obs_shape:
                raise ValueError("PPO requires an observation space with a shape.")
            obs_dim = int(obs_shape[0])
            action_dim = int(cast(Any, env.action_space).n)
            agent = PPOAgent(obs_dim, action_dim, device=device)

        agent.device = torch.device(device)
        agent.model.to(agent.device)

        timesteps = args.timesteps
        steps_per_rollout = 2048
        gamma = 0.99
        gae_lambda = 0.95
        total_steps = 0
        update = 0
        while total_steps < timesteps:
            obs_buf = []
            actions_buf = []
            logp_buf = []
            rewards_buf = []
            values_buf = []
            dones_buf = []

            obs, _ = env.reset()
            terminated = False
            truncated = False
            for t in range(min(steps_per_rollout, timesteps - total_steps)):
                action_t, logp_t, value_t = agent.sample_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(int(action_t.item()))
                obs_buf.append(obs)
                actions_buf.append(action_t.detach().cpu())
                logp_buf.append(logp_t.detach().cpu())
                values_buf.append(value_t.detach().cpu())
                rewards_buf.append(float(reward))
                dones_buf.append(bool(terminated or truncated))

                obs = next_obs if not (terminated or truncated) else env.reset()[0]
                total_steps += 1
                if total_steps >= timesteps:
                    break

            if terminated:
                last_value = 0.0
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    _, last_value_t = agent.model(obs_t)
                    last_value = float(last_value_t.squeeze(0).item())

            advantages, returns = compute_gae(
                rewards_buf,
                [float(v.item()) for v in values_buf],
                dones_buf,
                last_value,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            obs_tensor = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=agent.device)
            actions_tensor = torch.stack(actions_buf).to(device=agent.device)
            old_logp_tensor = torch.stack(logp_buf).to(device=agent.device)
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=agent.device)
            advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=agent.device)

            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std(unbiased=False) + 1e-8
            )

            agent.update(
                obs_tensor,
                actions_tensor,
                old_logp_tensor,
                returns_tensor,
                advantages_tensor,
            )

            update += 1
            if update % 1 == 0:
                print(f"Update {update}: total_steps={total_steps}")

        checkpoint_dir = Path(target_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "agent_trained.pt"
        agent.save(checkpoint_path)
        print(f"Saved trained agent to {checkpoint_path}")

        env.close()
        records = []
        return 0
    else:
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
