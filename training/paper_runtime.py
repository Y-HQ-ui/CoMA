"""Connect measured paper objectives, adaptive weights and generator token credit."""
from __future__ import annotations
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import numpy as np
import torch

from reward.paper_objectives import (AUDIO_NAMES, AUDIO_INITIAL_WEIGHTS, SYMBOLIC_NAMES, RewardProtocol,
                                    score_symbolic, score_audio, consistency, aggregate)
from reward.token_weights import WeightConfig, WeightController, HTTPWeightController
from training.token_ppo import temporal_credit


def discount_per_second(discount_per_unit, unit_seconds):
    if not np.isfinite([discount_per_unit, unit_seconds]).all() or not 0 < discount_per_unit <= 1 or unit_seconds <= 0:
        raise ValueError("positive discount unit and discount in (0,1] required")
    return float(discount_per_unit ** (1/unit_seconds))


class PaperRuntime:
    def __init__(self, stage, config_path, *, disable_consistency=False):
        self.family = stage.split("-")[0]
        self.config = json.loads(Path(config_path).read_text()) if config_path else {}
        self.protocol = RewardProtocol(**self.config.get("measurements", {}))
        self.undefined = self.config.get("undefined_correlation", "zero")
        if self.undefined not in {"zero", "error"}:
            raise ValueError("undefined_correlation must be zero or error")
        self.names = tuple(self.config.get("audio_components", AUDIO_NAMES)) if self.family == "audio" else SYMBOLIC_NAMES
        if disable_consistency:
            self.names = tuple(n for n in self.names if n != "consistency")
        if self.family == "audio" and (not self.names or not set(self.names) <= set(AUDIO_NAMES)):
            raise ValueError("unknown/empty audio components")
        self.llm = None
        if self.family == "audio":
            settings = WeightConfig(**self.config.get("weights", {}))
            self.llm = HTTPWeightController() if settings.mode == "llm" else None
            initial = self.config.get("initial_weights")
            if initial is None:
                total = sum(AUDIO_INITIAL_WEIGHTS[n] for n in self.names)
                initial = {n: AUDIO_INITIAL_WEIGHTS[n]/total for n in self.names}
            if initial and disable_consistency:
                initial = {k: v for k, v in initial.items() if k != "consistency"}
                total = sum(initial.values())
                initial = {k: v/total for k, v in initial.items()}
            self.controller = WeightController(self.names, settings, initial=initial, llm=self.llm)
        else:
            weights = self.config.get("symbolic_weights", dict(zip(SYMBOLIC_NAMES, [.30, .25, .20, .15, .10])))
            if set(weights) != set(SYMBOLIC_NAMES):
                raise ValueError("symbolic objective requires all five paper components")
            self.controller = WeightController(self.names, WeightConfig(), initial=weights)
        self.emotion_evaluator = None
        if "emotion" in self.names:
            plugin = self.config.get("emotion_evaluator")
            if not plugin:
                raise ValueError("full audio reward requires emotion_evaluator=module:function; explicitly select an ablation component subset otherwise")
            module, function = plugin.split(":")
            callback = getattr(importlib.import_module(module), function)
            options = self.config.get("emotion_evaluator_options", {})
            from functools import partial
            self.emotion_evaluator = partial(callback, **options)
        self.history = []

    def preflight(self, rows):
        from experiments.provenance import sha256_file
        checkpoint_hashes = {}
        for row in rows:
            if not row.get("reward_inputs"):
                raise ValueError(f"{row['sample_id']}: paper PPO requires measured reward_inputs JSON")
            evidence = self.inputs(row)
            if not evidence.get("measurement_source") or not evidence.get("visual"):
                raise ValueError("reward_inputs must contain measurement_source and real visual frame features")
            if self.family == "symbolic":
                if not isinstance(row.get("prefix"), str) or not row["prefix"].strip():
                    raise ValueError("paper symbolic PPO requires an explicit conditioned prefix, not an unconditioned ABC header")
                duration = row.get("duration_seconds")
                if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not np.isfinite(duration) or duration <= 0:
                    raise ValueError("symbolic phrase requires explicit positive duration_seconds")
            if self.family == "audio":
                guidance = row.get("symbolic_guidance")
                if not guidance or guidance.get("kind") != "frozen_policy_generation" or not guidance.get("checkpoint_sha256"):
                    raise ValueError("paper audio PPO requires frozen-policy symbolic_guidance provenance")
                if guidance.get("events_sha256") != sha256_file(row["symbolic_outputs"]):
                    raise ValueError("frozen symbolic guidance hash mismatch")
                if guidance.get("blueprint_sha256") != sha256_file(row["blueprint"]):
                    raise ValueError("frozen symbolic guidance blueprint hash mismatch")
                if guidance.get("sample_id") != row["sample_id"]:
                    raise ValueError("frozen symbolic guidance sample ID mismatch")
                # Older prepared manifests contain the immutable hash only.
                # New manifests also bind it to an existing checkpoint file.
                if guidance.get("checkpoint"):
                    checkpoint = str(Path(guidance["checkpoint"]).resolve())
                    if checkpoint not in checkpoint_hashes:
                        checkpoint_hashes[checkpoint] = sha256_file(checkpoint)
                    if checkpoint_hashes[checkpoint] != guidance["checkpoint_sha256"]:
                        raise ValueError("frozen symbolic guidance checkpoint hash mismatch")

    @staticmethod
    def inputs(row):
        return json.loads(Path(row["reward_inputs"]).read_text())

    def symbolic(self, events, row, duration, step):
        evidence = self.inputs(row)
        result = score_symbolic(events, evidence["visual"], duration, self.protocol)
        result["aggregation"] = aggregate(result["components"], dict(zip(self.names, self.controller.weights)), undefined=self.undefined)
        result.update(sample_id=row["sample_id"], step=step)
        self.history.append(result)
        return result["aggregation"]["reward"]

    def audio(self, events, row, path, duration, step, progress, mask, window_seconds):
        evidence = self.inputs(row)
        emotion = self.emotion_evaluator(audio_path=str(path), evidence=evidence) if self.emotion_evaluator else None
        result = score_audio(events, evidence["visual"], path, duration, self.protocol, emotion_measurement=emotion)
        observations = {}
        for name in self.names:
            value = result["components"][name]
            if value["status"] != "ok" and self.undefined == "error":
                raise ValueError(f"undefined {name}: {value.get('reason')}")
            observations[name] = value["reward"] if value["status"] == "ok" else 0.
        weights, update = self.controller.update(step, observations, progress=progress, human_feedback=evidence.get("human_feedback"))
        result["aggregation"] = aggregate(result["components"], weights, undefined=self.undefined)
        curves = result["evidence"].pop("audio_window_curves")
        times = np.asarray(curves["times"])
        windows, details = [], []
        for start in np.arange(0, duration, window_seconds):
            end = min(duration, start+window_seconds)
            selected = (times >= start) & (times < end)
            local = dict(result["components"])
            if "consistency" in self.names:
                local["consistency"] = consistency(*[np.asarray(curves[name])[selected] for name in
                    ("symbolic_onset", "audio_onset", "symbolic_energy", "audio_energy")], self.protocol)
            measured = aggregate(local, weights, undefined=self.undefined)
            reward = measured["reward"]*(end-start)/duration
            windows.append((start, end, reward))
            details.append({"start": float(start), "end": float(end), "reward_mass": reward,
                            "local_consistency": local["consistency"], "zeroed": measured["zeroed_unavailable"]})
        token_times = torch.arange(mask.shape[1], device=mask.device)/50.
        intervals = torch.stack((token_times.clamp_max(duration), (token_times+.02).clamp_max(duration)), -1)[None]
        rewards = temporal_credit([windows], intervals, mask)
        elapsed = intervals[..., 1]-intervals[..., 0]
        result.update(sample_id=row["sample_id"], step=step, weight_update=update, windows=details,
                      assigned_reward_mass=float(rewards.sum()),
                      global_components_scope="clip-level measurements; consistency recomputed per acoustic window")
        self.history.append(result)
        return rewards, elapsed

    def metadata(self):
        return {"version": "paper_runtime_v1", "measurements": asdict(self.protocol),
                "active_components": list(self.names), "weights": asdict(self.controller.config),
                "undefined_correlation": self.undefined, "emotion_evaluator": self.config.get("emotion_evaluator"),
                "full_six_component_audio_reward": self.family == "audio" and set(self.names) == set(AUDIO_NAMES)}

    def flush(self, root):
        root = Path(root)
        (root/"reward_trace.json").write_text(json.dumps(self.history, indent=2, allow_nan=False))
        (root/"weight_trajectory.json").write_text(json.dumps({"protocol": self.metadata(), "rows": self.controller.history}, indent=2))
        (root/"weight_state.json").write_text(json.dumps(self.controller.state_dict(), indent=2))
        if self.llm is not None:
            (root/"weight_llm_trace.json").write_text(json.dumps(self.llm.traces, indent=2))
