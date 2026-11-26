import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    action_noises: torch.Tensor | None = None


class RLAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        hidden_dim: int = 256,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dim=hidden_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }
        self._pending_noise = []

    def _distribution(self, states: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.policy_net(states)
        return torch.distributions.Categorical(logits=logits)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        logit_noise_scale: float = 0.0,
    ) -> Tuple[int, float, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(state_tensor)
        noise = torch.zeros_like(logits)
        if logit_noise_scale > 0.0 and not deterministic:
            noise = float(logit_noise_scale) * torch.randn_like(logits)
            logits = logits + noise
        self._pending_noise.append(noise.detach().squeeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state_tensor).squeeze(-1)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - float(dones[t])
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
            advantages[t] = last_adv
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def build_rollout_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        log_probs: List[float],
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        advantages, returns = self.compute_gae(rewards, values, dones, last_value, gamma, gae_lambda)
        if len(advantages)>1 and advantages.std()>1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        noises = torch.stack(self._pending_noise[-len(states):]) if len(self._pending_noise)>=len(states) else torch.zeros(len(states),self.action_dim,device=self.device)
        self._pending_noise.clear()
        return RolloutBatch(
            states=torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(actions, dtype=torch.long, device=self.device),
            old_log_probs=torch.as_tensor(log_probs, dtype=torch.float32, device=self.device),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            action_noises=noises,
        )

    def update_policy(
        self,
        batch: RolloutBatch,
        *,
        ppo_clip: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        update_epochs: int,
        minibatch_size: int,
    ) -> Dict[str, float]:
        num_samples = batch.states.shape[0]
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        for _ in range(update_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                states_mb = batch.states[idx]
                actions_mb = batch.actions[idx]
                old_log_probs_mb = batch.old_log_probs[idx]
                returns_mb = batch.returns[idx]
                adv_mb = batch.advantages[idx]

                logits = self.policy_net(states_mb)
                if batch.action_noises is not None:
                    logits = logits + batch.action_noises[idx]
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()
                values = self.value_net(states_mb).squeeze(-1)

                ratio = torch.exp(new_log_probs - old_log_probs_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, returns_mb)

                self.policy_optimizer.zero_grad()
                (policy_loss - entropy_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                (value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
                self.value_optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy.item())

        self.training_stats["policy_loss"].append(last_policy_loss)
        self.training_stats["value_loss"].append(last_value_loss)
        self.training_stats["entropy"].append(last_entropy)
        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
        }

    def save(self, filepath: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
