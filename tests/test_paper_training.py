import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from reward.paper_objectives import (RewardProtocol, aggregate, consistency, key_stability,
    note_diversity, score_symbolic, score_audio, symbolic_components)
from reward.token_weights import WeightConfig, WeightController
from training.paper_runtime import PaperRuntime, discount_per_second


class PaperTrainingTests(unittest.TestCase):
    def visual(self):
        return {"times": [0., .5, 1., 1.5], "motion_intensity": [0., 1., .2, .8],
                "cut_flags": [0, 1, 0, 0], "visual_novelty": [.1, .7, .3, .8]}

    def events(self):
        return [{"track_id": "piano", "pitch": p, "onset": t, "duration": .25, "velocity": v}
                for p, t, v in [(60, 0., .4), (64, .5, .8), (67, 1., .6), (72, 1.5, .9)]]

    def test_raw_equations_and_fixed_entropy_normalizers(self):
        uniform = key_stability(np.ones(24)/24)
        self.assertAlmostEqual(uniform["raw"], 1-np.log(24))
        self.assertAlmostEqual(uniform["reward"], 0.)
        self.assertAlmostEqual(key_stability([1., 0.])["reward"], 1.)
        diversity = note_diversity([60, 62, 60, 62], [.25, .5, .25, .5], RewardProtocol())
        self.assertAlmostEqual(diversity["raw"], 2*np.log(2))
        # Adding support categories does not renormalize this sample to 1.
        self.assertLess(diversity["reward"], .2)

    def test_consistency_is_correlation_not_dtw_or_onset_f1(self):
        curve = np.array([0., 1., .2, .8, .3])
        item = consistency(curve, curve, curve, curve, RewardProtocol())
        self.assertAlmostEqual(item["raw"], 1.)
        inverse = consistency(curve, 1-curve, curve, 1-curve, RewardProtocol())
        self.assertAlmostEqual(inverse["reward"], 0.)
        undefined = consistency(np.ones(5), np.ones(5), curve, curve, RewardProtocol())
        self.assertEqual(undefined["status"], "unavailable")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            aggregate({"consistency": undefined}, {"consistency": 1.})
        self.assertEqual(aggregate({"consistency": undefined}, {"consistency": 1.}, undefined="zero")["reward"], 0.)

    def test_symbolic_eq12_seconds_lag_and_boundary_one_to_one(self):
        visual = np.zeros(200); visual[[30, 80, 130]] = [1., .7, .5]
        symbolic = np.roll(visual, 5)
        values = symbolic_components(pitches=[60, 64, 67], durations=[.25]*3,
            key_posterior=[1., 0.], violations=1, visual_curve=visual, symbolic_curve=symbolic,
            visual_boundaries=[.5, .55], phrase_boundaries=[.52], protocol=RewardProtocol())
        self.assertAlmostEqual(values["beat"]["reward"], 1.)
        self.assertAlmostEqual(values["phrase"]["raw"], 2/3)
        self.assertAlmostEqual(values["playability"]["reward"], 2/3)
        result = score_symbolic(self.events(), self.visual(), 2.)
        self.assertEqual(set(result["components"]), {"beat", "phrase", "key", "diversity", "playability"})

    def test_fixed_smoothing_is_exact_identity(self):
        a = WeightController(["motion", "consistency"], WeightConfig(mode="fixed", rho=1.))
        for step in range(10):
            weights, trace = a.update(step, {"motion": .1, "consistency": .8}, progress=step/9)
            self.assertEqual(weights, {"motion": .5, "consistency": .5})
            self.assertEqual(trace["l2_step"], 0.)
        with self.assertRaises(ValueError):
            WeightConfig(mode="fixed", consistency_final_gain=2.)

    def test_adaptation_obeys_floor_sum_step_and_llm_error_is_explicit(self):
        cfg = WeightConfig(mode="rule", rho=1., max_step=.02, floor=.1, update_every=1)
        a = WeightController(["motion", "consistency"], cfg)
        for step in range(30):
            weights, trace = a.update(step, {"motion": .01, "consistency": .99})
            self.assertAlmostEqual(sum(weights.values()), 1.)
            self.assertGreaterEqual(min(weights.values()), .1-1e-12)
            self.assertLessEqual(trace["l2_step"], .02+1e-12)
        bad = WeightController(["motion", "consistency"], WeightConfig(mode="llm", update_every=1), llm=lambda _: {"logits": {"wrong": 9}})
        weights, trace = bad.update(0, {"motion": .1, "consistency": .5})
        self.assertEqual(trace["fallback"], "hold_previous_coefficients")
        self.assertEqual(weights, {"motion": .5, "consistency": .5})
        restored = WeightController(a.names, cfg)
        restored.load_state_dict(a.state_dict())
        np.testing.assert_array_equal(restored.weights, a.weights)

    def test_per_window_discount_conversion(self):
        gamma = discount_per_second(.8, .4)
        self.assertAlmostEqual(gamma**.4, .8)
        self.assertAlmostEqual(gamma**.02, .8**(.02/.4))
        self.assertAlmostEqual(discount_per_second(.95, 2.)**2, .95)
        with self.assertRaises(ValueError):
            discount_per_second(.8, 0.)

    def test_real_waveform_reward_credit_and_consistency_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 16000
            times = np.arange(2*sr)/sr
            waveform = np.zeros_like(times)
            for event in self.events():
                age = times-event["onset"]
                waveform += .1*event["velocity"]*np.sin(2*np.pi*440*age)*np.exp(-20*np.maximum(age, 0))*(age>=0)*(age<.25)
            path = root/"audio.wav";sf.write(path, waveform, sr)
            measurements = score_audio(self.events(), self.visual(), path, 2.)
            self.assertEqual(measurements["components"]["emotion"]["status"], "unavailable")
            self.assertEqual(measurements["components"]["consistency"]["status"], "ok")
            config = root/"config.json"
            config.write_text(json.dumps({"audio_components": ["motion", "tonal", "consistency"],
                                          "weights": {"mode": "rule", "update_every": 1}}))
            evidence = root/"evidence.json"
            evidence.write_text(json.dumps({"measurement_source": "synthetic_fixture", "visual": self.visual()}))
            runtime = PaperRuntime("audio-ppo", str(config))
            row = {"sample_id": "synthetic", "reward_inputs": str(evidence)}
            mask = torch.ones(1, 103, dtype=torch.bool)
            reward, elapsed = runtime.audio(self.events(), row, path, 2., 0, 0., mask, .4)
            self.assertEqual(reward.shape, mask.shape)
            self.assertAlmostEqual(elapsed.sum().item(), 2., places=5)
            self.assertEqual(reward[0, 100:].sum().item(), 0.)
            self.assertAlmostEqual(reward.sum().item(), sum(w["reward_mass"] for w in runtime.history[0]["windows"]), places=5)
            runtime.flush(root)
            self.assertEqual(len(json.loads((root/"reward_trace.json").read_text())), 1)
            ablated = PaperRuntime("audio-ppo", str(config), disable_consistency=True)
            self.assertNotIn("consistency", ablated.names)
            self.assertAlmostEqual(ablated.controller.weights.sum(), 1.)

    def test_no_fabricated_full_emotion_backend(self):
        with self.assertRaisesRegex(ValueError, "emotion_evaluator"):
            PaperRuntime("audio-ppo", "")

    def test_frozen_guidance_binds_sample_blueprint_events_and_checkpoint(self):
        from experiments.provenance import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/"config.json"
            config.write_text(json.dumps({"audio_components": ["consistency"]}))
            evidence = root/"evidence.json"
            evidence.write_text(json.dumps({"measurement_source": "unit_test", "visual": self.visual()}))
            paths = {key: root/(key+".json") for key in ("blueprint", "events", "checkpoint")}
            for key, path in paths.items():
                path.write_text(key)
            row = {"sample_id": "one", "reward_inputs": str(evidence),
                   "blueprint": str(paths["blueprint"]), "symbolic_outputs": str(paths["events"]),
                   "symbolic_guidance": {"kind": "frozen_policy_generation", "sample_id": "one",
                       "checkpoint": str(paths["checkpoint"]),
                       **{key+"_sha256": sha256_file(value) for key, value in paths.items()}}}
            runtime = PaperRuntime("audio-ppo", config)
            runtime.preflight([row])
            symbolic_runtime = PaperRuntime("symbolic-ppo", config)
            symbolic_row = {"sample_id": "one", "reward_inputs": str(evidence), "prefix": "K:C<n>"}
            with self.assertRaisesRegex(ValueError, "duration_seconds"):
                symbolic_runtime.preflight([symbolic_row])
            symbolic_runtime.preflight([{**symbolic_row, "duration_seconds": 2.}])
            for key, path in paths.items():
                original = path.read_text()
                path.write_text("changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    runtime.preflight([row])
                path.write_text(original)
            row["sample_id"] = "other"
            with self.assertRaisesRegex(ValueError, "sample ID"):
                runtime.preflight([row])

    def test_audio_ppo_never_reads_target_waveform_or_encodes_labels(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from training.data import audio_example, read_training_manifest
        class TextInputs(dict):
            def to(self, device):
                return self
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbolic = root/"events.json"
            symbolic.write_text(json.dumps([{"segment_id": 0, "events": self.events()}]))
            blueprint = Path("examples/blueprint_smoke.json").resolve()
            manifest = root/"train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "one", "split": "train",
                "blueprint": str(blueprint), "symbolic_outputs": str(symbolic)}))
            row = read_training_manifest(manifest, "audio-ppo")[0]
            processor = lambda **kwargs: TextInputs(input_ids=torch.tensor([[1, 2]]))
            with patch("soundfile.read", side_effect=AssertionError("PPO read target audio")):
                payload, _, _ = audio_example(row, SimpleNamespace(), processor, "cpu", supervised=False)
            self.assertEqual(set(payload), {"features", "text_inputs"})
            self.assertEqual(payload["features"].shape, (1, 100, 29))


if __name__ == "__main__":
    unittest.main()
