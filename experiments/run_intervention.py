import argparse
import json
from pathlib import Path
from planning import load_blueprint_json
from planning.timeline import merge_overlapping_segments
from .interventions import intervene, score_change
from .realization import RealizationBackend
from reward.refinement import GenerationBudget

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blueprint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--segment-id', type=int, required=True)
    p.add_argument('--track-id')
    p.add_argument('--field', required=True)
    p.add_argument('--value-json', required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--mupt-model-dir', default='')
    p.add_argument('--musicgen-model-dir', default='')
    p.add_argument('--mupt-device', default='cpu')
    p.add_argument('--musicgen-device', default='cpu')
    p.add_argument('--adapter-checkpoint', default='')
    p.add_argument('--symbolic-policy-checkpoint', default='')
    p.add_argument('--symbolic-max-new-tokens', type=int, default=384)
    args = p.parse_args(argv)
    base = merge_overlapping_segments(load_blueprint_json(args.blueprint))
    changed = intervene(base, segment_id=args.segment_id, track_id=args.track_id, field=args.field, value=json.loads(args.value_json))
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('use a new output directory')
    backend = RealizationBackend(str(root), args.mupt_model_dir, args.musicgen_model_dir, args.mupt_device, args.musicgen_device, args.adapter_checkpoint, symbolic_policy_checkpoint=args.symbolic_policy_checkpoint, max_new_tokens=args.symbolic_max_new_tokens)
    original = backend.generate(base, seed=args.seed, budget=GenerationBudget(10000))
    counterfactual = backend.generate(changed, seed=args.seed, budget=GenerationBudget(10000))
    segment = next((s for s in base.segments if s.segment_id == args.segment_id))
    before = [e for item in original.symbolic_outputs for e in item['events']]
    after = [e for item in counterfactual.symbolic_outputs for e in item['events']]
    report = {'intervention': vars(args), 'base': original.artifact_dir, 'changed': counterfactual.artifact_dir, 'track_changes': {t.track_id or t.role: score_change(before, after, track_id=t.track_id or t.role, start=segment.start_time, end=segment.end_time) for t in segment.tracks}}
    import soundfile as sf
    import numpy as np
    a, sr = sf.read(original.audio_manifest['soundtrack_path'])
    b, sr2 = sf.read(counterfactual.audio_manifest['soundtrack_path'])
    if a.shape != b.shape or sr != sr2:
        raise ValueError('intervention output time axes differ')
    mask = np.ones(len(a), dtype=bool)
    mask[round(segment.start_time * sr):round(segment.end_time * sr)] = False
    report['non_target_waveform_mae'] = float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else None
    report['limitation'] = 'Control realization includes disclosed pitch-range/density postprocessing; text-only rendering cannot prove score fidelity.'
    (root / 'intervention.json').write_text(json.dumps(report, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
