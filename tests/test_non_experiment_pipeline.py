import json
import tempfile
import unittest
from pathlib import Path

from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_controls_to_prompts
from planning.blueprint_parser import blueprint_from_segment_priors, load_blueprint_json
from run_coma_pipeline import run_pipeline_from_blueprint


class NonExperimentPipelineTest(unittest.TestCase):
    def _blueprint(self):
        return blueprint_from_segment_priors(
            {
                "prior_5d": [
                    {
                        "start_time": 0.0,
                        "end_time": 4.0,
                        "motion_intensity": 0.2,
                        "cut_density": 0.1,
                        "brightness": 0.7,
                        "color_warmth": 0.8,
                        "visual_novelty": 0.2,
                    }
                ]
            }
        )

    def test_symbolic_guidance_is_attached_to_prompt(self):
        blueprint = self._blueprint()
        controls = blueprint_to_render_controls(
            blueprint,
            symbolic_outputs=[
                {
                    "segment_id": 0,
                    "abc": 'X:1<n>L:1/8<n>Q:120<n>M:4/4<n>K:Cmaj<n>|:"C" CDEFGABc |',
                }
            ],
        )
        prompt = render_controls_to_prompts(controls)[0]
        self.assertIn("symbolic guidance:", prompt)

    def test_rl_refinement_is_json_serializable(self):
        blueprint = self._blueprint()
        with tempfile.TemporaryDirectory() as output_dir:
            run_pipeline_from_blueprint(
                blueprint=blueprint,
                output_dir=output_dir,
                refine_with_rl=True,
                refine_with_hierarchical_rl=False,
                rl_episodes=1,
                enable_repair_cycle=True,
                critic_feedback_mode="llm_or_human",
                human_feedback=None,
                llm_feedback=None,
            )
            payload = json.loads((Path(output_dir) / "rl_refinement.json").read_text())
            self.assertEqual(payload["num_episodes"], 1)

    def test_blueprint_round_trip_preserves_track_section_role(self):
        blueprint = self._blueprint()
        with tempfile.TemporaryDirectory() as output_dir:
            path = Path(output_dir) / "blueprint.json"
            path.write_text(json.dumps(blueprint.to_dict()), encoding="utf-8")
            loaded = load_blueprint_json(str(path))
            self.assertEqual(
                loaded.segments[0].tracks[0].section_role,
                blueprint.segments[0].tracks[0].section_role,
            )


if __name__ == "__main__":
    unittest.main()
