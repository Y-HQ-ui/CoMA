import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from planning import load_blueprint_json
from reward.critic import diagnose_failure
from reward.emotion import aligned_correlation
from reward.paper_objectives import AUDIO_INITIAL_WEIGHTS
from reward.refinement import Candidate, GenerationBudget, refine_inference
from reward.token_weights import WeightController, WeightConfig


class ComaRevisionTests(unittest.TestCase):
    def test_weights_wait_eight_rollouts_and_use_paper_initial_vector(self):
        controller = WeightController(tuple(AUDIO_INITIAL_WEIGHTS), WeightConfig(mode="rule"), initial=AUDIO_INITIAL_WEIGHTS)
        for i in range(8):
            weights, trace = controller.update(i, {k: .2 for k in AUDIO_INITIAL_WEIGHTS})
            self.assertEqual(trace["updated"], i == 7)
            if i < 7:
                self.assertEqual(weights, AUDIO_INITIAL_WEIGHTS)
        self.assertAlmostEqual(sum(weights.values()), 1.)
        self.assertLessEqual(trace["l2_step"], .1)

    def candidate(self):
        return Candidate(load_blueprint_json("examples/blueprint_smoke.json"), [], {},
                         {"R_vis": .9, "R_mus": .9, "R_cons": .9, "R_total": .9})

    def test_fixed_proposal_count_ignores_satisfied_thresholds_and_patience(self):
        def generate(scope, current, seed, budget):
            budget.consume("test")
            return current
        for count in (5, 30):
            final, trace = refine_inference(self.candidate(), generate, budget=GenerationBudget(None),
                                           stopping="fixed", max_iterations=count)
            self.assertEqual(trace["proposals"], count)
            self.assertEqual(trace["acceptances"], 0)
            self.assertEqual(trace["calls"], count)

    def test_convergence_and_incomplete_budget_preserve_incumbent(self):
        def generate(scope, current, seed, budget):
            budget.consume("test")
            return current
        initial = self.candidate()
        final, trace = refine_inference(initial, generate, budget=GenerationBudget(None),
                                       stopping="convergence", max_iterations=60, patience=5)
        self.assertEqual(trace["proposals"], 5)
        self.assertEqual(trace["stop"], "no_improvement_patience")
        final, trace = refine_inference(initial, generate, budget=GenerationBudget(2),
                                       stopping="fixed", max_iterations=5)
        self.assertEqual(trace["stop"], "generation_call_budget")
        self.assertEqual(final.rewards, initial.rewards)
        self.assertTrue(trace["history"][-1]["incomplete"])

    def test_six_component_routing_uses_worst_individual_deficit(self):
        rewards = {k: .8 for k in AUDIO_INITIAL_WEIGHTS}
        rewards.update(consistency=.1, R_total=.6)
        diagnosis = diagnose_failure(rewards, global_threshold=.55, component_threshold=.45)
        self.assertEqual(diagnosis["source_level"], "audio")
        rewards["motion"] = .1
        self.assertEqual(diagnose_failure(rewards, global_threshold=.55, component_threshold=.45)["source_level"], "blueprint")

    def test_va_uses_zero_lag_and_independent_video_targets(self):
        times = [1.5, 2., 2.5, 3.]
        values = [[.1, .4], [.3, .1], [.2, .3], [.4, .2]]
        target = {"times": times, "values": values, "source": "test_only", "used_target_audio": False}
        score, count = aligned_correlation(times, values, target)
        self.assertAlmostEqual(score, 1.)
        self.assertEqual(count, 4)
        bad = copy.deepcopy(target)
        bad["used_target_audio"] = True
        with self.assertRaisesRegex(ValueError, "silent-video"):
            aligned_correlation(times, values, bad)
        with self.assertRaisesRegex(ValueError, "constant"):
            aligned_correlation(times, np.ones((4, 2)), target)

    def test_paper_inference_uses_fixed_weights_without_training_controller(self):
        from reward.paper_scoring import PaperArtifactScorer
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"evidence.json"
            path.write_text(json.dumps({"visual": {"test": True}, "measurement_source": "test"}))
            components = {k: {"status": "ok", "reward": i/10, "raw": i/10} for i, k in enumerate(AUDIO_INITIAL_WEIGHTS)}
            with patch("reward.emotion.evaluate_musicnn", return_value={}), patch("reward.paper_scoring.score_audio", return_value={"components": components}):
                scorer = PaperArtifactScorer(path, "configs/paper_training_full.json")
                result = scorer(self.candidate().blueprint, [{"events": []}], {"soundtrack_path": "test.wav"})
            expected = sum(AUDIO_INITIAL_WEIGHTS[k]*v["reward"] for k, v in components.items())
            self.assertAlmostEqual(result["rewards"]["R_total"], expected)
            self.assertEqual(result["aggregation"]["weights"], AUDIO_INITIAL_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
