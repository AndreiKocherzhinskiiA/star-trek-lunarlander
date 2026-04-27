# Start Trek

Simple LunarLander baseline project for the EPITECH Start Trek reinforcement-learning assignment.

## What is included

- `train.py`: runs a baseline job from a YAML config
- `eval.py`: evaluates a baseline config for a chosen number of episodes
- `reproduce.py`: reruns baseline configs
- `baseline.py`: random and heuristic policies with CSV logging and optional video recording
- `random.yaml` / `heuristic.yaml`: example baseline configs

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run a baseline

Random policy:

```bash
python train.py --config configs/baseline/random.yaml
```

Heuristic policy:

```bash
python train.py --config configs/baseline/heuristic.yaml
```

## Evaluate

```bash
python eval.py --config configs/baseline/heuristic.yaml --episodes 100
```

## Reproduce baseline artifacts

```bash
python reproduce.py
```

## Outputs

Each run writes artifacts to its configured output directory:

- `config.json`
- `episodes.csv`
- `summary.csv`
- `videos/` when video recording is enabled

## Notes

This repository currently focuses on baseline instrumentation. The full learning agent, ablations, and final report are still to be added.
