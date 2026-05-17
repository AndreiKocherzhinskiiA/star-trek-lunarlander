# Start Trek

LunarLander reinforcement-learning project for the EPITECH Start Trek assignment.

## Current status

The repository now contains:

- baseline policies: `random`, `heuristic`
- a trainable PPO agent
- deterministic evaluation for PPO checkpoints
- config-driven training and evaluation
- reproducible seed control for train/eval runs
- saved benchmark artifacts under `results/`

## Recommended checkpoint

For evaluation, use:

- config: `configs/agent/ppo_refine_conservative.yaml`
- checkpoint: `results/ppo_refine_conservative/agent_best.pt`

This is the verified winning checkpoint. Do not use the final weights from the same run by default; `agent_trained.pt` regressed after the best validation point.

## Verified result

The winning checkpoint was re-evaluated over 100 episodes with deterministic PPO inference:

```bash
python eval.py --config configs/agent/ppo_refine_conservative.yaml --episodes 100 --agent-path results/ppo_refine_conservative/agent_best.pt
```

Observed result:

- mean return: `207.39`
- safe landings: `94 / 100`
- out of view: `1 / 100`
- sleep terminations: `5 / 100`

This clears the assignment target of mean score `>= 200` over 100 consecutive episodes.

## Reproducibility protocol

The repo now supports explicit seed overrides:

```bash
python train.py --config <config> --train --seed 0
python eval.py --config <config> --episodes 100 --seed 0
```

Seed control covers:

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA when available

One-command benchmark regeneration:

```bash
python scripts/reproduce.py
```

Default behavior:

- evaluates `random`, `heuristic`, early PPO, and winning PPO
- runs seeds `0..4`
- uses `100` episodes per evaluation
- regenerates per-run `config.json`, `episodes.csv`, `summary.csv`
- regenerates per-run SVG plots from `episodes.csv`
- writes aggregate artifacts under `results/repro_benchmark/`

Aggregate repro outputs:

- `benchmark_summary.csv`
- `benchmark_summary.md`
- `benchmark_summary.svg`

## Five-seed benchmark

Benchmark setup:

- seeds: `0..4`
- episodes per eval: `100`
- metric: `mean_return`
- 95% CI computed as `mean +/- 1.96 * std / sqrt(5)`

Comparison table:

| Method | Artifact used | Mean return | Std | 95% CI |
|---|---|---:|---:|---:|
| Random | `configs/baseline/random.yaml` | -186.20 | 7.83 | [-193.06, -179.34] |
| Heuristic | `configs/baseline/heuristic.yaml` | 86.52 | 1.75 | [84.99, 88.05] |
| Early PPO | `configs/baseline/ppo.yaml` + `configs/agent/agent_trained.pt` | 105.84 | 1.31 | [104.69, 106.99] |
| Winning PPO | `configs/agent/ppo_refine_conservative.yaml` + `results/ppo_refine_conservative/agent_best.pt` | 207.17 | 0.28 | [206.93, 207.42] |

Interpretation:

- the heuristic baseline is much stronger than random, but still far from the target
- the first PPO checkpoint improved on the heuristic baseline, but not enough
- the conservative PPO refinement is the first setup that clears the target consistently across seeds

## Main commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a baseline:

```bash
python train.py --config configs/baseline/random.yaml
python train.py --config configs/baseline/heuristic.yaml
```

Train PPO:

```bash
python train.py --config configs/agent/ppo_refine_conservative.yaml --train
```

Evaluate PPO:

```bash
python eval.py --config configs/agent/ppo_refine_conservative.yaml --episodes 100 --agent-path results/ppo_refine_conservative/agent_best.pt
```

Rebuild benchmark artifacts:

```bash
python scripts/reproduce.py
```

## Key implementation notes

- PPO evaluation is deterministic.
- Best-checkpoint selection is validation-based during training.
- YAML configs control PPO hyperparameters and validation cadence.
- The baseline runner still writes `config.json`, `episodes.csv`, and `summary.csv` for each run.
- the strongest PPO run uses `agent_best.pt`, not the final checkpoint
- deterministic evaluation was necessary to avoid understating PPO performance
- the five-seed benchmark is the evidence for robustness, not a single cherry-picked run
