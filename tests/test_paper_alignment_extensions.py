from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import main as main_module
from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.reference_audio import extract_reference_audio_features
from metrics import evaluate_paper_metrics
from planning.blueprint_parser import blueprint_from_segment_priors
from reward.evidence import RewardEvidenceEvaluator
from reward.hierarchical_env import HierarchicalCoMAEnv


class PaperAlignmentExtensionTests(unittest.TestCase):
    def _blueprint(self):
        return blueprint_from_segment_priors(
            {
                "prior_5d": [
                    {
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "motion_intensity": 0.7,
                        "cut_density": 0.5,
                        "brightness": 0.6,
                        "color_warmth": 0.7,
                        "visual_novelty": 0.4,
                    }
                ]
            }
        )

    def _audio(self, path: Path, duration: float = 2.0) -> None:
        sample_rate = 16000
        time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
        waveform = 0.2 * np.sin(2.0 * np.pi * 220.0 * time)
        for start in range(0, waveform.size, sample_rate // 2):
            waveform[start : start + 120] += np.hanning(min(120, waveform.size - start)).astype(np.float32)
        sf.write(path, waveform, sample_rate)

    def test_composer_requires_llm_configuration(self):
        controller = main_module.BlueprintComposerController()
        with mock.patch.object(main_module, "BASE_URL", ""), mock.patch.object(main_module, "API_KEY", ""), mock.patch.object(main_module, "MODEL_NAME", "MODEL_NAME_NOT_SET"):
            with self.assertRaises(RuntimeError):
                controller.plan_blueprint({"prior_5d": []})

    def test_reference_audio_features_are_llm_ready(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "reference.wav"
            self._audio(path)
            features = extract_reference_audio_features(str(path), temporal_hz=2.0)
        self.assertEqual(features["source"], "reference_audio_waveform")
        self.assertEqual(len(features["trajectories"]["energy"]), 4)
        self.assertGreaterEqual(features["tempo_bpm"], 50.0)
        self.assertIn("waveform", features)

    def test_audio_policy_uses_subsecond_windows(self):
        blueprint = self._blueprint()
        controls = blueprint_to_render_controls(blueprint, window_seconds=0.4)
        self.assertEqual(len(controls[0]["render_windows"]), 5)
        env = HierarchicalCoMAEnv(
            blueprint,
            reward_evaluator=RewardEvidenceEvaluator(),
            audio_window_seconds=0.4,
        )
        env.reset()
        env.step(0, [0, 1, 2, 3, 4])
        self.assertEqual(len(env.audio_action_history), 5)
        self.assertEqual(env.audio_action_history[-1]["window_index"], 4)

    def test_paper_metrics_report_backends_and_missing_requirements(self):
        blueprint = self._blueprint()
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_path = Path(temporary_dir) / "generated.wav"
            self._audio(audio_path)
            result = evaluate_paper_metrics(
                blueprint=blueprint,
                audio_manifest={
                    "segment_paths": [str(audio_path)],
                    "soundtrack_path": str(audio_path),
                },
                reference_audio_path=str(audio_path),
            )
        metrics = result["metrics"]
        # Pooled blueprint values cannot stand in for measured visual events.
        self.assertEqual(metrics["rhythm_control"]["status"], "unavailable")
        # A two-second clip contains only one default two-second segment.
        self.assertEqual(metrics["chroma_diversity"]["status"], "unavailable")
        # A style reference does not turn free generation into reconstruction.
        self.assertEqual(metrics["si_sdr"]["status"], "unavailable")
        self.assertEqual(metrics["imagebind_score"]["status"], "unavailable")
        self.assertEqual(metrics["frechet_audio_distance"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
