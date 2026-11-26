import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from planning.blueprint_parser import blueprint_from_segment_priors
from reward.config import build_default_rl_config
from reward.critic import build_critic_decision
from reward.evidence import RewardEvidenceEvaluator, extract_audio_features, extract_symbolic_features
from reward.hierarchical_refine import refine_blueprint_hierarchical
from reward.llm_controller import LLMWeightController
from reward.rewards import MultiSourceRewardAggregator, SymbolicAudioConsistencyReward, SymbolicRewardCalculator
from vedio.prior_feature_pack import extract_prior_5d


class CoreAlignmentTest(unittest.TestCase):
    def _blueprint(self):
        return blueprint_from_segment_priors(
            {
                "prior_5d": [
                    {
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "motion_intensity": 0.2,
                        "cut_density": 0.1,
                        "brightness": 0.7,
                        "color_warmth": 0.8,
                        "visual_novelty": 0.2,
                    },
                    {
                        "start_time": 1.0,
                        "end_time": 3.0,
                        "motion_intensity": 0.7,
                        "cut_density": 0.5,
                        "brightness": 0.4,
                        "color_warmth": 0.3,
                        "visual_novelty": 0.7,
                    },
                ]
            }
        )

    def test_reward_aggregation_spans_unit_interval(self):
        config = build_default_rl_config()
        aggregator = MultiSourceRewardAggregator(config)
        self.assertAlmostEqual(aggregator.compute_visual_rewards(1.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(aggregator.compute_musicality_rewards(1.0, 1.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(aggregator.aggregate_rewards(1.0, 1.0, 1.0), 1.0)
        self.assertEqual(SymbolicRewardCalculator(config).compute_phrase_consistency([], []), 1.0)
        self.assertEqual(
            SymbolicAudioConsistencyReward(config).compute_onset_alignment(np.zeros(8), np.zeros(8)),
            1.0,
        )

    def test_real_symbolic_and_waveform_evidence(self):
        symbolic = extract_symbolic_features(
            "X:1\nL:1/8\nK:C\n| C z D2 |",
            duration=2.0,
            key="C",
        )
        self.assertEqual(symbolic["source"], "abc")
        self.assertEqual(len(symbolic["pitches"]), 2)
        self.assertEqual(len(symbolic["durations"]), 2)
        self.assertEqual(int(np.count_nonzero(symbolic["onset"])), 2)

        with tempfile.TemporaryDirectory() as temporary_dir:
            sample_rate = 16000
            waveform = np.zeros(sample_rate, dtype=np.float32)
            waveform[1000:1100] = 1.0
            waveform[8000:8100] = -1.0
            audio_path = Path(temporary_dir) / "transients.wav"
            sf.write(audio_path, waveform, sample_rate)
            audio = extract_audio_features(str(audio_path))

        self.assertEqual(audio["source"], "waveform")
        self.assertGreater(float(np.max(audio["onset"])), 0.9)

        evidence = RewardEvidenceEvaluator().evaluate(self._blueprint(), 0, {})
        self.assertEqual(evidence["temporal_alignment_hz"], 2.0)
        self.assertEqual(len(evidence["symbolic_data"]["beat"]["g_sym"]), 4)

    def test_human_critic_overrides_llm_and_rule_fallback_is_explicit(self):
        class FakeCritic:
            def diagnose(self, **kwargs):
                return {
                    "repair_required": True,
                    "source_level": "audio",
                    "dominant_failure": "consistency",
                    "confidence": 0.9,
                    "rationale": "audio mismatch",
                }

        class BrokenCritic:
            def diagnose(self, **kwargs):
                raise RuntimeError("offline")

        rewards = {"R_total": 0.2, "R_vis": 0.2, "R_mus": 0.2, "R_cons": 0.1, "R_sym": 0.2}
        decision = build_critic_decision(
            rewards,
            feedback_mode="llm_or_human",
            global_threshold=0.55,
            component_threshold=0.45,
            max_repair_passes=2,
            human_feedback={"veto": True},
            llm_critic=FakeCritic(),
        )
        self.assertEqual(decision.feedback_source, "human")
        self.assertEqual(decision.repair_plan["scope"], "none")

        fallback = build_critic_decision(
            rewards,
            feedback_mode="llm",
            global_threshold=0.55,
            component_threshold=0.45,
            max_repair_passes=2,
            llm_critic=BrokenCritic(),
        )
        self.assertEqual(fallback.feedback_source, "rule")
        self.assertEqual(fallback.diagnosis["llm_fallback"], "rule_based")

    def test_llm_controller_changes_cross_group_weight_mass(self):
        config = build_default_rl_config()
        config.use_llm_controller = True
        config.llm_update_freq = 1
        config.tau = 1.0
        config.delta = 10.0
        controller = LLMWeightController(
            config,
            lambda **kwargs: {
                "alpha_vis": [10.0, 10.0, 10.0],
                "beta_mus": [1.0, 1.0, 1.0, 1.0],
                "gamma_cons": 1.0,
            },
        )
        alpha_vis, beta_mus, gamma_cons = controller.update_weights(0.1, 0.8, 0.8, step=0)
        self.assertAlmostEqual(float(alpha_vis.sum() + beta_mus.sum() + gamma_cons), 1.0)
        self.assertGreater(float(alpha_vis.sum()), float(beta_mus.sum()))

    def test_hierarchical_warmup_then_joint_schedule(self):
        config = build_default_rl_config()
        config.symbolic_warmup_episodes = 1
        config.update_symbolic_during_joint = False
        config.hidden_dim = 16
        config.update_epochs = 1
        config.batch_size = 8
        config.llm_update_freq = 99
        config.use_llm_controller = False
        result = refine_blueprint_hierarchical(
            self._blueprint(),
            config=config,
            num_episodes=2,
            reward_evaluator=RewardEvidenceEvaluator(),
            device="cpu",
        )
        history = result["training_history"]
        self.assertEqual(history["training_phase"], ["symbolic_warmup", "joint"])
        self.assertTrue(history["symbolic_ppo_stats"][0])
        self.assertEqual(history["symbolic_ppo_stats"][1], {})
        self.assertEqual(history["audio_ppo_stats"][0], {})
        self.assertTrue(history["audio_ppo_stats"][1])
        self.assertEqual(history["audio_episode_rewards"][0], 0.0)

    def test_video_segments_use_fifty_percent_overlap(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            video_path = str(Path(temporary_dir) / "tiny.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
            if not writer.isOpened():
                self.skipTest("OpenCV video writer unavailable")
            for frame_index in range(50):
                frame = np.full((48, 64, 3), frame_index * 5 % 255, dtype=np.uint8)
                cv2.circle(frame, (frame_index % 64, 24), 5, (255, 0, 0), -1)
                writer.write(frame)
            writer.release()
            payload = extract_prior_5d(
                video_path,
                segment_duration=2.0,
                segment_overlap=0.5,
                sample_fps=2.0,
            )

        starts = [round(item["start_time"], 3) for item in payload["prior_5d"]]
        ends = [round(item["end_time"], 3) for item in payload["prior_5d"]]
        self.assertEqual(starts, [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(ends, [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(payload["summary"]["segment_stride"], 1.0)


if __name__ == "__main__":
    unittest.main()
