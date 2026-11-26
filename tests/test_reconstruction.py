import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from planning.blueprint_parser import blueprint_from_segment_priors,MusicBlueprintParser
from planning.timeline import merge_overlapping_segments,TempoMap
from symbolic.events import abc_to_events,export_midi
from audio.timeline import assemble_timeline,replace_audio_interval
from audio.temporal_control import build_temporal_features,FEATURES
from audio.abc_to_render_controls import blueprint_to_render_controls
from reward.refinement import Candidate,GenerationBudget,refine_inference
from reward.config import build_default_rl_config
from reward.llm_controller import LLMWeightController
from metrics.protocol import lagged_ncc,imagebind_score,inception_score,frechet_embeddings,onset_f1
from experiments.statistics import paired_video_bootstrap,crossed_human_bootstrap,holm_correction
from experiments.data_protocol import audit_dataset
from experiments.export_results import normalized_radar
from experiments.interventions import intervene,score_change


def blueprint():
    return blueprint_from_segment_priors({"prior_5d":[{"start_time":0.,"end_time":2.,"motion_intensity":.2},
                                                     {"start_time":1.,"end_time":3.,"motion_intensity":.8}]})


class ReconstructionTests(unittest.TestCase):
    def test_overlap_has_one_timeline(self):
        b=blueprint();out=merge_overlapping_segments(b)
        self.assertEqual([(s.start_time,s.end_time) for s in out.segments],[(0,1),(1,2),(2,3)])
        self.assertAlmostEqual(out.segments[1].energy,(b.segments[0].energy+b.segments[1].energy)/2)
        self.assertEqual(len(b.segments),2)

    def test_invalid_intervals_and_duplicate_ids_fail(self):
        raw=blueprint().to_dict();raw["segments"][1]["segment_id"]=0
        with self.assertRaises(ValueError):MusicBlueprintParser().parse(raw)
        raw["segments"][1]["segment_id"]=1;raw["segments"][1]["end_time"]=float("nan")
        with self.assertRaises(ValueError):MusicBlueprintParser().parse(raw)

    def test_tempo_integrates_cross_boundary_notes(self):
        t=TempoMap([(0,120),(1,60)])
        self.assertEqual(t.seconds_to_beats(2),3)
        self.assertEqual(t.beats_to_seconds(3),2)
        events=abc_to_events("X:1\nL:1/4\nM:4/4\nK:C\nC3",tempo_map=t)
        self.assertEqual(events[0].duration,2)

    def test_abc_chords_accidentals_voices_and_ties(self):
        notes=abc_to_events("X:1\nM:4/4\nL:1/8\nQ:1/4=120\nK:G\nV:1\nF2 [GB]2 C2-C2 |\nV:2\nG,8 |")
        self.assertEqual(len(notes),5)
        self.assertIn(66,[n.pitch for n in notes])
        self.assertEqual(len({n.track_id for n in notes}),2)
        tied=next(n for n in notes if n.pitch==60)
        self.assertAlmostEqual(tied.duration,1.)
        self.assertEqual(len([n for n in notes if n.onset==.5]),2)

    def test_midi_time_roundtrip(self):
        import mido
        tempo=TempoMap([(0,120),(1,60)])
        events=abc_to_events("X:1\nL:1/4\nK:C\nC3",tempo_map=tempo)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"score.mid";export_midi(events,p,tempo);m=mido.MidiFile(p)
            self.assertAlmostEqual(m.length,2.,places=5)

    def test_mupt_synchronized_voices_and_empty_delimiters(self):
        from symbolic.notation import decode_mupt
        abc = 'X:1<n>M:4/4<n>L:1/8<n>Q:1/4=120<n>K:C<n> <|> C8 | G,8 <|> <|> D8 | A,8 <|>'
        standard, audit = decode_mupt(abc)
        self.assertEqual(audit['voices'], 2)
        self.assertEqual(audit['bar_blocks'], 2)
        notes = abc_to_events(abc)
        self.assertEqual([e.onset for e in notes], [0., 0., 2., 2.])
        self.assertEqual(len({e.track_id for e in notes}), 2)
        self.assertNotIn('<|>', standard)

    def test_mupt_missing_voice_is_rest_and_bad_alignment_rejected(self):
        prefix = 'X:1<n>M:4/4<n>L:1/8<n>Q:1/4=120<n>K:C<n>'
        notes = abc_to_events(prefix + '<|> C8 | G,8 <|> D8 <|> E8 | A,8 <|>')
        self.assertEqual([e.onset for e in notes if e.track_id == 'voice:2'], [0., 4.])
        with self.assertRaises(ValueError):
            abc_to_events(prefix + '<|> C4 | G,8 <|> D8 | A,8 <|>')
        repeated=abc_to_events(prefix + '<|> |: <|> <|> |: C8 | <|> D8 :| <|>')
        self.assertEqual(len({e.track_id for e in repeated}),1)
        self.assertEqual([e.onset for e in repeated],[0.,2.])

    def test_audio_duration_and_mask_preservation(self):
        waveform=assemble_timeline([np.ones(200),np.ones(200)],[(0,2),(1,3)],100,fade_seconds=0)
        self.assertEqual(waveform.shape,(300,));self.assertTrue(np.all(waveform==1))
        original=np.arange(200,dtype=float);replaced=replace_audio_interval(original,np.zeros(200),start_sample=30,end_sample=70,fade_samples=5)
        np.testing.assert_array_equal(original[:30],replaced[:30]);np.testing.assert_array_equal(original[70:],replaced[70:])
        with self.assertRaises(ValueError):assemble_timeline([np.ones(10)],[(0,2)],100)

    def test_adapter_features_follow_controls(self):
        b=merge_overlapping_segments(blueprint());controls=blueprint_to_render_controls(b)
        f=build_temporal_features(controls,[{"pitch":60,"onset":.5,"duration":1.,"velocity":.7}])
        self.assertEqual(f.shape,(150,len(FEATURES)))
        self.assertEqual(f[25,7],1);self.assertGreater(f[60,10],0)
        self.assertEqual(f[20,7],0)

    def test_weight_simplex_floor_bound_and_static_noop(self):
        cfg=build_default_rl_config();cfg.epsilon=.02;cfg.tau=.9;cfg.delta=.01;cfg.use_llm_controller=True;cfg.llm_update_freq=1
        controller=LLMWeightController(cfg,lambda **k:{"alpha_vis":[1e12,0,0],"beta_mus":[0]*4,"gamma_cons":0})
        def vector():return np.r_[controller.alpha_vis,controller.beta_mus,controller.gamma_cons]
        prev=vector()
        for i in range(30):
            controller.update_weights(.1,.2,.3,step=i);now=vector()
            self.assertAlmostEqual(now.sum(),1.,places=12);self.assertGreaterEqual(now.min(),cfg.epsilon-1e-12)
            self.assertLessEqual(np.linalg.norm(now-prev),cfg.delta+1e-8);prev=now
        cfg.weight_mode="fixed";controller.update_weights(.1,.2,.3,step=31);np.testing.assert_array_equal(prev,vector())

    def test_invalid_weight_proposals_fail(self):
        from reward.weight_constraints import simplex_with_floor
        with self.assertRaises(ValueError):simplex_with_floor([1,float("nan")])

    def test_no_constant_perfect_ncc_or_truncated_embeddings(self):
        with self.assertRaises(ValueError):lagged_ncc([1,1,1],[1,1,1],sample_hz=100)
        self.assertAlmostEqual(lagged_ncc([0,1,0,0],[0,0,1,0],sample_hz=10,max_lag_seconds=.1)[0],1)
        with self.assertRaises(ValueError):imagebind_score([1,0],[1,0,0])
        self.assertAlmostEqual(imagebind_score([1,0],[1,0]),1)

    def test_metric_contracts(self):
        self.assertAlmostEqual(inception_score([[1,0],[0,1]]),2)
        with self.assertRaises(ValueError):inception_score([[.5,.8],[.2,.3]])
        with self.assertRaises(ValueError):frechet_embeddings([[0,1]],[[0,1]])
        self.assertLess(frechet_embeddings([[0,1],[1,0]],[[0,1],[1,0]]),1e-6)
        self.assertEqual(onset_f1([0,1],[0,0.01,1]),.8)

    def test_frozen_refinement_rejects_worse_and_accounts_budget(self):
        b=blueprint();r={"R_total":.2,"R_vis":.8,"R_mus":.8,"R_cons":.1,"R_sym":.8}
        initial=Candidate(b,[{"abc":"score"}],{},r)
        calls=[]
        def regenerate(scope,current,seed,budget):
            budget.consume("musicgen");calls.append(scope)
            current.rewards={**r,"R_total":.1};return current
        final,trace=refine_inference(initial,regenerate,budget=GenerationBudget(1),max_iterations=5)
        self.assertEqual(final.rewards["R_total"],.2);self.assertEqual(calls,["audio"])
        self.assertEqual(trace["calls"],1);self.assertFalse(trace["history"][0]["accepted"])
        self.assertEqual(trace["stop"],"generation_call_budget")

    def test_audio_repair_cannot_modify_upstream(self):
        b=blueprint();r={"R_total":.1,"R_vis":.8,"R_mus":.8,"R_cons":.1,"R_sym":.8}
        def corrupt(scope,current,seed,budget):
            budget.consume("musicgen");current.blueprint.segments[0].energy=.9;current.rewards["R_total"]=.9;return current
        _,trace=refine_inference(Candidate(b,[],{},r),corrupt,budget=GenerationBudget(1),max_iterations=1)
        self.assertIn("preserved blueprint",trace["history"][0]["error"])

    def test_human_veto_stops_frozen_refinement(self):
        regenerate=mock.Mock()
        initial=Candidate(blueprint(),[],{},{'R_total':.1})
        _,trace=refine_inference(initial,regenerate,budget=GenerationBudget(5),
            decision_callback=lambda _: {'human_veto':True},strategy='full')
        regenerate.assert_not_called()
        self.assertEqual(trace['stop'],'human_veto')

    def test_leakage_and_pairing(self):
        rows=[{"sample_id":"a","split":"train","dataset":"A","movie_id":"same","music_work_id":"one","input_condition":"V"},
              {"sample_id":"b","split":"test","dataset":"B","movie_id":"same","music_work_id":"two","input_condition":"V"}]
        self.assertFalse(audit_dataset(rows)["valid"])
        left={("a",1):2,("a",2):2,("b",1):3};right={k:v-1 for k,v in left.items()}
        result=paired_video_bootstrap(left,right,replicates=100)
        self.assertEqual(result["ci95"],[1,1]);self.assertEqual(result["n_independent_videos"],2)
        with self.assertRaises(ValueError):paired_video_bootstrap(left,{("a",1):2})

    def test_human_cluster_and_holm(self):
        rows=[{"participant_id":p,"video_id":v,"dimension":"quality","method":m,"rating":r} for p in ["p1","p2"] for v in ["v1","v2"] for m,r in [("a",4),("b",3)]]
        result=crossed_human_bootstrap(rows,"a","b",replicates=100)
        self.assertEqual(result["ci95"],[1,1]);self.assertEqual(holm_correction([.01,.04,.03]),[.03,.06,.06])

    def test_radar_preserves_order(self):
        data={"rows":[{"method":"GVMGen","metrics":{"IS":{"mean":3.26}}},{"method":"FilmComposer","metrics":{"IS":{"mean":3.24}}}]}
        result=normalized_radar(data,{"IS":"higher"})
        self.assertGreater(result["values"]["GVMGen"][0],result["values"]["FilmComposer"][0])

    def test_single_control_intervention(self):
        b=blueprint();out=intervene(b,segment_id=0,field="density",value=.1,track_id="bass")
        raw=b.to_dict();edited=out.to_dict();edited["segments"][0]["tracks"][1]["density"]=raw["segments"][0]["tracks"][1]["density"]
        self.assertEqual(raw,edited)


if __name__=="__main__":unittest.main()
