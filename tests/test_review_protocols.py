import json
from pathlib import Path
import tempfile
import unittest

from experiments.listening_study import design, analyze
from experiments.weight_report import summarize
from experiments.statistics import paired_video_signflip, holm_correction


class ReviewProtocolTests(unittest.TestCase):
    def test_weight_table_uses_run_equal_means_and_rejects_sum_130(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol={"active_components":["a","consistency"],"weights":{"floor":.01}}
            a={"protocol":protocol,"rows":[{"step":0,"weights":{"a":.2,"consistency":.8}}]}
            b={"protocol":protocol,"rows":[{"step":i,"weights":{"a":.8,"consistency":.2}} for i in range(3)]}
            paths=[root/'a.json',root/'b.json']
            for p,v in zip(paths,[a,b]):p.write_text(json.dumps(v))
            result=summarize(paths)
            self.assertAlmostEqual(result['summary']['a']['mean_run_equal'],.5)
            self.assertAlmostEqual(sum(x['mean_run_equal'] for x in result['summary'].values()),1)
            with self.assertRaisesRegex(ValueError,'duplicate trajectory'):
                summarize([paths[0],paths[0]])
            b['rows'][0]['weights']['consistency']=.5
            paths[1].write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'simplex'):summarize(paths)

    def test_prospective_study_counts_pairing_and_incomplete_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['v1.mp4','v2.mp4','a.wav','b.wav']:(root/name).write_bytes(name.encode())
            spec={'study_id':'synthetic_test_only','participants':['p1','p2'], 'videos_per_participant':2,
                'dimensions':['VA','AQ'],'instructions':'test form only','seed':5,
                'videos':[{'video_id':v,'video':v+'.mp4','audio':{'A':'a.wav','B':'b.wav'}} for v in ['v1','v2']]}
            path=root/'spec.json';path.write_text(json.dumps(spec))
            key=design(path,root/'study')
            self.assertEqual(key['expected_audiovisual_trials'],8)
            self.assertEqual(key['expected_dimension_ratings'],16)
            forms=sorted((root/'study').glob('participant_*.json'))
            self.assertNotIn('method',forms[0].read_text())
            with self.assertRaisesRegex(ValueError,'missing/out-of-range'):
                analyze(root/'study/private_allocation_key.json',forms,'A','B',replicates=20)
            # Synthetic numbers stay inside this unit test, never research artifacts.
            lookup={(r['participant_id'],r['stimulus_id']):r for r in key['key']}
            for p in forms:
                form=json.loads(p.read_text())
                for trial in form['trials']:
                    method=lookup[(form['participant_id'],trial['stimulus_id'])]['method']
                    trial['ratings']={d:4 if method=='A' else 3 for d in spec['dimensions']}
                p.write_text(json.dumps(form))
            result=analyze(root/'study/private_allocation_key.json',forms,'A','B',replicates=20)
            self.assertEqual(result['counted_dimension_ratings'],16)
            self.assertEqual(result['statistics']['VA']['mean_difference'],1)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                analyze(root/'study/private_allocation_key.json',forms[:1],'A','B',replicates=20)

    def test_seed_repeats_do_not_inflate_independent_units(self):
        left={(video,seed):1. for video in ['v1','v2','v3'] for seed in range(5)}
        right={key:0. for key in left}
        result=paired_video_signflip(left,right)
        self.assertEqual(result['n_videos'],3)
        self.assertEqual(result['randomizations'],8)
        self.assertEqual(result['p_value'],.25)
        self.assertEqual(holm_correction([.01,.04,.03]),[.03,.06,.06])

    def test_mmp_cannot_bypass_audio_transcription(self):
        from metrics.uniform_transcription import evaluate_records,select_measurement
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audio.wav';path.write_bytes(b'synthetic_unit_test_only')
            row={'sample_id':'test','method':'A','seed':1,'audio':str(path)}
            protocol={'transcriber_checkpoint':'test-transcriber','evaluator_checkpoint':'test-evaluator',
                'prompt':'test rubric','decoding':{'temperature':0},'score_definition':'synthetic unit-test score'}
            calls=[]
            def transcribe(**kwargs):
                calls.append(kwargs['audio_path']);return {'abc':'X:1\nK:C\nC','checkpoint':kwargs['checkpoint']}
            def evaluate(**kwargs):return {'score':.5,'checkpoint':kwargs['checkpoint']}
            report=evaluate_records([row],protocol,transcribe,evaluate)
            self.assertEqual(calls,[str(path)])
            self.assertEqual(select_measurement(report,path)[0],.5)
            with self.assertRaisesRegex(ValueError,'bypass'):
                evaluate_records([{**row,'abc':'clean_score'}],protocol,transcribe,evaluate)
            path.write_bytes(b'changed_audio')
            with self.assertRaisesRegex(ValueError,'exact waveform'):
                select_measurement(report,path)

    def test_review_audit_checks_links_and_does_not_certify_science(self):
        from experiments.audit_review import audit, markdown
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'evidence.py').write_text('# synthetic audit fixture')
            definitions={key:key for key in ['implemented','reconstructed','partial','requires_original_evidence']}
            items=[{'id':identifier,'source':'unit-test fixture','requirement':'check link',
                    'implementation_status':'partial','implementation':['evidence.py'],
                    'verification':[],'remaining':'synthetic fixture only'}
                   for identifier in [*[f'R{n}.1' for n in range(1,12)],*[f'P{n}' for n in range(10,19)]]]
            matrix={'scope':'coverage only','status_definitions':definitions,'items':items,'sources':{'paper':'evidence.py'}}
            path=root/'matrix.json';path.write_text(json.dumps(matrix))
            report=audit(path,root)
            self.assertEqual(report['requirements'],20)
            self.assertIn('does not run tests',report['scope'])
            self.assertIn('R11.1',markdown(matrix))
            (root/'evidence.py').unlink()
            with self.assertRaisesRegex(ValueError,'missing'):
                audit(path,root)


if __name__=='__main__':unittest.main()
