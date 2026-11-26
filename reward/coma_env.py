from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from audio.abc_to_render_controls import blueprint_to_render_controls
from planning.blueprint_schema import MusicBlueprint
from .evidence import RewardEvidenceEvaluator


@dataclass(frozen=True)
class ActionSpec:
    name: str
    tempo_delta: int
    energy_delta: float
    brightness_delta: float
    texture_bias: float


DEFAULT_ACTION_SPECS: List[ActionSpec] = [
    ActionSpec("keep", 0, 0.00, 0.00, 0.00),
    ActionSpec("tempo_up", 8, 0.05, 0.00, 0.05),
    ActionSpec("tempo_down", -8, -0.05, 0.00, -0.05),
    ActionSpec("energy_up", 0, 0.10, 0.05, 0.10),
    ActionSpec("energy_down", 0, -0.10, -0.05, -0.10),
    ActionSpec("brighten", 0, 0.00, 0.10, 0.05),
    ActionSpec("darken", 0, 0.00, -0.10, -0.05),
    ActionSpec("accent_transition", 4, 0.08, 0.00, 0.12),
]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class CoMARLEnv:
    def __init__(
        self,
        blueprint: MusicBlueprint,
        action_specs: List[ActionSpec] | None = None,
        reward_evaluator: RewardEvidenceEvaluator | None = None,
    ):
        self.blueprint = blueprint
        self.base_controls = blueprint_to_render_controls(blueprint)
        self.action_specs = action_specs or DEFAULT_ACTION_SPECS
        self.reward_evaluator = reward_evaluator or RewardEvidenceEvaluator()
        self.index = 0
        self.current_controls: List[Dict[str, Any]] = []

    @property
    def action_dim(self) -> int:
        return len(self.action_specs)

    @property
    def state_dim(self) -> int:
        return 18

    def reset(self) -> np.ndarray:
        self.index = 0
        self.current_controls = [dict(control) for control in self.base_controls]
        return self._state_from_index(self.index)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        spec = self.action_specs[action]
        control = self.current_controls[self.index]
        control["tempo_bpm"] = int(np.clip(control["tempo_bpm"] + spec.tempo_delta, 70, 170))
        control["energy"] = _clip01(control["energy"] + spec.energy_delta)
        control["brightness_hint"] = _clip01(
            control["video_prior_5d"]["brightness"] + spec.brightness_delta
        )
        control["texture_bias"] = spec.texture_bias

        reward_info = self.reward_evaluator.evaluate(self.blueprint, self.index, control)
        visual_values = list(reward_info["visual_data"].values())
        musical_values = list(reward_info["musicality_data"].values())
        reward = _clip01(0.5 * float(np.mean(visual_values)) + 0.5 * float(np.mean(musical_values)))
        reward_info["R_total"] = reward

        self.index += 1
        done = self.index >= len(self.current_controls)
        next_state = np.zeros(self.state_dim, dtype=np.float32) if done else self._state_from_index(self.index)

        info = {
            "action_name": spec.name,
            "render_control": dict(control),
            "symbolic_data": reward_info["symbolic_data"],
            "consistency_data": reward_info["consistency_data"],
            "visual_data": reward_info["visual_data"],
            "musicality_data": reward_info["musicality_data"],
            "reward_breakdown": reward_info,
            "evidence_source": reward_info.get("evidence_source", {}),
        }
        return next_state, reward, done, info

    def _state_from_index(self, index: int) -> np.ndarray:
        control = self.current_controls[index]
        prior = control["video_prior_5d"]
        section_map = {
            "intro": 0.1,
            "build": 0.3,
            "sustain": 0.5,
            "chorus": 0.7,
            "climax": 0.9,
            "outro": 0.2,
        }
        state = np.array(
            [
                prior["motion_intensity"],
                prior["cut_density"],
                prior["brightness"],
                prior["color_warmth"],
                prior["visual_novelty"],
                control["energy"],
                control["tempo_bpm"] / 180.0,
                1.0 if control["meter"] == "4/4" else 0.5,
                section_map.get(control["section"], 0.4),
                1.0 if "melody" in control["track_roles"] else 0.0,
                1.0 if "bass" in control["track_roles"] else 0.0,
                1.0 if "pad" in control["track_roles"] else 0.0,
                len(control["chords"]) / 4.0,
                index / max(1, len(self.current_controls) - 1),
                (len(self.current_controls) - index) / max(1, len(self.current_controls)),
                control.get("texture_bias", 0.0) + 0.5,
                control["start_time"] / max(1e-6, self.current_controls[-1]["end_time"]),
                control["end_time"] / max(1e-6, self.current_controls[-1]["end_time"]),
            ],
            dtype=np.float32,
        )
        return state
