from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .config import RLConfig


class RepairTriggerNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TriggerPrediction:
    trigger_prob: float
    heuristic_score: float
    blended_score: float
    action_confidence: float
    uncertainty: float
    human_override: float
    deficit_sum: float
    policy_trigger: bool


def heuristic_trigger_components(
    config: RLConfig,
    critic_decision: Any,
    *,
    confidence: float,
) -> Dict[str, float]:
    deficits = critic_decision.diagnosis.get("deficits", {})
    deficit_sum = float(sum(max(0.0, float(value)) for value in deficits.values()))
    uncertainty = 1.0 - float(np.clip(confidence, 0.0, 1.0))
    human_override = 0.0
    if critic_decision.feedback_source == "human":
        human_override += 0.12
        if critic_decision.repair_plan.get("scope") != "none":
            human_override += 0.08
    score = (
        config.repair_trigger_base
        + config.repair_trigger_deficit_weight * deficit_sum
        + config.repair_trigger_uncertainty_weight * uncertainty
        + human_override
    )
    return {
        "score": float(score),
        "deficit_sum": deficit_sum,
        "action_confidence": float(np.clip(confidence, 0.0, 1.0)),
        "uncertainty": uncertainty,
        "human_override": human_override,
        "triggered": bool(score >= 1.0),
    }


def build_trigger_features(
    critic_decision: Any,
    *,
    confidence: float,
) -> np.ndarray:
    deficits = critic_decision.diagnosis.get("deficits", {})
    scope = critic_decision.repair_plan.get("scope", "none")
    source = critic_decision.feedback_source
    return np.asarray(
        [
            float(deficits.get("visual", 0.0)),
            float(deficits.get("musicality", 0.0)),
            float(deficits.get("consistency", 0.0)),
            float(deficits.get("symbolic", 0.0)),
            float(np.clip(confidence, 0.0, 1.0)),
            1.0 - float(np.clip(confidence, 0.0, 1.0)),
            1.0 if scope == "blueprint" else 0.0,
            1.0 if scope == "symbolic" else 0.0,
            1.0 if scope == "audio" else 0.0,
            1.0 if source == "human" else 0.0,
            1.0 if source == "llm" else 0.0,
        ],
        dtype=np.float32,
    )


class RepairTriggerPolicy:
    def __init__(self, config: RLConfig, *, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = RepairTriggerNetwork(input_dim=11, hidden_dim=config.repair_trigger_hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.repair_trigger_lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.training_history: list[Dict[str, float]] = []

    def predict(self, critic_decision: Any, *, confidence: float) -> TriggerPrediction:
        heuristic = heuristic_trigger_components(self.config, critic_decision, confidence=confidence)
        features = build_trigger_features(critic_decision, confidence=confidence)
        with torch.no_grad():
            logits = self.model(torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0))
            trigger_prob = float(torch.sigmoid(logits).item())
        heuristic_prob = float(np.clip(heuristic["score"] / 1.5, 0.0, 1.0))
        alpha = float(np.clip(self.config.repair_trigger_blend_alpha, 0.0, 1.0))
        blended = alpha * trigger_prob + (1.0 - alpha) * heuristic_prob
        return TriggerPrediction(
            trigger_prob=trigger_prob,
            heuristic_score=heuristic["score"],
            blended_score=blended,
            action_confidence=heuristic["action_confidence"],
            uncertainty=heuristic["uncertainty"],
            human_override=heuristic["human_override"],
            deficit_sum=heuristic["deficit_sum"],
            policy_trigger=bool(blended >= 0.5),
        )

    def update(
        self,
        critic_decision: Any,
        *,
        confidence: float,
        target: bool,
        outcome_bonus: float = 0.0,
    ) -> Dict[str, float]:
        features = build_trigger_features(critic_decision, confidence=confidence)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        y = torch.as_tensor([1.0 if target else 0.0], dtype=torch.float32, device=self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        if outcome_bonus != 0.0:
            loss = loss * (1.0 + float(np.clip(outcome_bonus, -0.5, 1.0)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        stats = {
            "loss": float(loss.item()),
            "target": float(y.item()),
            "outcome_bonus": float(outcome_bonus),
        }
        self.training_history.append(stats)
        return stats
