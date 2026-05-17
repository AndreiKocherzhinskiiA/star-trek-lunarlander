from __future__ import annotations
import argparse
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from start_trek.baseline import run_policy, make_env, build_policy
from start_trek.config import load_config
from start_trek.agent import PPOAgent, PPOPolicy
from start_trek.repro import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Start Trek policy config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes.")
    parser.add_argument("--output-dir", help="Override eval output directory.")
    parser.add_argument("--agent-path", help="Path to saved agent checkpoint (for PPO).")
    parser.add_argument("--seed", type=int, help="Override run seed from config.")

    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, run=replace(config.run, seed=args.seed))
    seed_everything(config.run.seed)

    run_config = replace(config.run, episodes=args.episodes)
    eval_config = replace(config, run=run_config)
    output_dir = Path(args.output_dir) if args.output_dir else config.run.output_dir / "eval"

    probe_config = replace(config, video=replace(config.video, enabled=False))
    env = make_env(probe_config, output_dir=None)
    policy = build_policy(
        config.policy.name,
        env.observation_space,
        env.action_space,
        agent_path=args.agent_path,
    )
    env.close()

    if isinstance(policy, PPOAgent):
        policy = PPOPolicy(policy.model)

    records = run_policy(eval_config, output_dir=output_dir, policy=policy)
    mean_return = sum(record.return_ for record in records) / len(records)
    print(f"Evaluated {len(records)} episodes. Mean return: {mean_return:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
