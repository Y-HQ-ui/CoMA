import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf

from planning import load_blueprint_json
from reward.refinement import Candidate,GenerationBudget
from experiments.realization import RealizationBackend
from metrics.protocol import evaluate_review_metrics

ROOT=Path(__file__).resolve().parents[1]


class IntegrationContractTests(unittest.TestCase):
    def fixture(self,root):
        blueprint=load_blueprint_json(ROOT/'examples/blueprint_smoke.json')
        times=np.arange(64000)/32000
        original=.1*np.sin(2*np.pi*220*times)*(1+.3*np.cos(2*np.pi*times))
        path=root/'original.wav';sf.write(path,original,32000)
        symbolic=[{'segment_id':0,'events':[{'track_id':'piano','pitch':60,'onset':.2,'duration':.3,'velocity':.7},
            {'track_id':'bass','pitch':48,'onset':1.2,'duration':.4,'velocity':.7}]}]
        manifest={'soundtrack_path':str(path),'segment_paths':[str(path)],'sample_rate':32000}
        return blueprint,symbolic,manifest

    def test_local_repair_preserves_saved_non_target_samples_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);blueprint,symbolic,manifest=self.fixture(root)
            before,_=sf.read(manifest['soundtrack_path'])
            class Renderer:
                def render_one(self,prompt,**kw):
                    sf.write(kw['output_path'],np.zeros(round(kw['duration']*32000)),32000)
            backend=RealizationBackend(str(root/'repair'),local_audio_window_seconds=.4)
            budget=GenerationBudget(1)
            with patch('audio.musicgen_renderer.MusicGenRendererSession.get',return_value=Renderer()):
                candidate=backend.regenerate('audio',Candidate(blueprint,symbolic,manifest,{}),7,budget)
            after,_=sf.read(candidate.audio_manifest['soundtrack_path']);mask=candidate.audio_manifest['local_repair']
            left,right=mask['start_sample'],mask['end_sample']
            np.testing.assert_array_equal(before[:left],after[:left]);np.testing.assert_array_equal(before[right:],after[right:])
            self.assertGreater(np.abs(before[left:right]-after[left:right]).sum(),0)
            self.assertEqual(candidate.symbolic_outputs,symbolic);self.assertEqual(budget.calls,1)

    def test_dv_checks_hash_leakage_and_hcs_records_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);blueprint,symbolic,manifest=self.fixture(root)
            spec={'chroma':{'segment_seconds':1.,'frames_per_segment':8},
                  'hcs':{'sigma_tc':1.,'scale_source':'explicit_unit_test_scale'},
                  'dynamic_variation':{'reference_audio':manifest['soundtrack_path'],'reference_role':'held_out_evaluation'}}
            report=evaluate_review_metrics(blueprint=blueprint,symbolic_outputs=symbolic,audio_manifest=manifest,metric_inputs=spec)
            self.assertEqual(report['metrics']['harmonic_consistency']['status'],'ok')
            self.assertAlmostEqual(report['metrics']['dynamic_variation']['value'],0)
            duplicate=root/'copied_reference.wav';duplicate.write_bytes(Path(manifest['soundtrack_path']).read_bytes())
            with self.assertRaisesRegex(ValueError,'duplicates the conditioning'):
                evaluate_review_metrics(blueprint=blueprint,audio_manifest=manifest,metric_inputs=spec,reference_audio_path=str(duplicate))

    def test_training_manifest_rejects_test_split(self):
        from training.data import read_training_manifest
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'samples.jsonl';path.write_text(json.dumps({'sample_id':'test','split':'test','abc':'irrelevant.abc'})+'\n')
            with self.assertRaisesRegex(ValueError,'split=train'):read_training_manifest(path,'symbolic-sft')

    def test_table_rejects_different_evaluator_settings(self):
        from experiments.export_results import aggregate_runs,COMPARISON_FIELDS
        protocol={key:'' for key in COMPARISON_FIELDS}
        protocol.update(sample_ids=['s'],seeds=[1],evaluation_settings={'max_lag_seconds':.1})
        first={'method':'a','protocol':protocol,'records':[{'sample_id':'s','seed':1,'status':'ok','metrics':{'RC':{'status':'ok','backend':'test','value':.5}}}]}
        second=copy.deepcopy(first);second['method']='b';second['protocol']['evaluation_settings']['max_lag_seconds']=.5
        with self.assertRaisesRegex(ValueError,'incompatible table protocols'):aggregate_runs([first,second],['RC'])

    def test_pipeline_evaluates_accepted_final_audio(self):
        from run_coma_pipeline import run_pipeline_from_blueprint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);blueprint,symbolic,manifest=self.fixture(root)
            replacement=dict(manifest,soundtrack_path='accepted_candidate.wav')
            rewards={'R_total':.1,'R_vis':.8,'R_mus':.8,'R_sym':.8,'R_cons':.1}
            def regenerate(self,scope,current,seed,budget):
                budget.consume('musicgen');current.audio_manifest=replacement;current.rewards=dict(rewards,R_total=.9);return current
            with patch('run_coma_pipeline.generate_blueprint_score',return_value=symbolic),\
                 patch('run_coma_pipeline.render_musicgen_prompts',return_value=manifest),\
                 patch('model_registry.resolve_model',return_value='/offline/test_model'),\
                 patch('reward.artifact_scoring.score_artifacts',return_value={'rewards':rewards}),\
                 patch('reward.llm_critic_client.LLMStructuredCritic.from_config',return_value=None),\
                 patch.object(RealizationBackend,'regenerate',regenerate),\
                 patch('run_coma_pipeline.evaluate_paper_metrics',return_value={}) as evaluate:
                run_pipeline_from_blueprint(blueprint=blueprint,output_dir=str(root/'run'),refine_with_rl=False,
                    refine_with_hierarchical_rl=False,rl_episodes=0,enable_repair_cycle=True,critic_feedback_mode='rule',
                    human_feedback=None,llm_feedback=None,generate_symbolic=True,generate_audio=True,
                    inference_iterations=1,evaluate_metrics=True)
            self.assertEqual(evaluate.call_args.kwargs['audio_manifest']['soundtrack_path'],'accepted_candidate.wav')
            saved=json.loads((root/'run/final_audio_manifest.json').read_text())
            self.assertEqual(saved,replacement)


if __name__=='__main__':unittest.main()
