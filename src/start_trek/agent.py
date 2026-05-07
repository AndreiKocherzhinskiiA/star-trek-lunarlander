from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.model(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    def sample_action(self, observation: np.ndarray):
        """Return action, log_prob, and value tensors for a single observation (no grad)."""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def get_action_and_value(self, observations: torch.Tensor, actions: torch.Tensor):
        logits, values = self.model(observations)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        *,
        epochs: int = 4,
        minibatch_size: int = 64,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        """Perform PPO update on given batch tensors (already on correct device).

        obs: (N, obs_dim)
        actions: (N,)
        old_log_probs: (N,)
        returns: (N,)
        advantages: (N,)
        """
        N = obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, minibatch_size):
                idx = perm[start : start + minibatch_size]
                obs_mb = obs[idx]
                actions_mb = actions[idx]
                old_lp_mb = old_log_probs[idx]
                returns_mb = returns[idx]
                adv_mb = advantages[idx]

                logits, values = self.model(obs_mb)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_lp_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), returns_mb)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "obs_dim": self.model.shared[0].in_features,
                "action_dim": self.model.policy_head.out_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PPOAgent":
        checkpoint = torch.load(path, map_location=device)
        agent = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            device=device,
        )
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return agent

class PPOPolicy:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1)
        return int(action.item())