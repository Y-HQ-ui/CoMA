from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

def generate_symbolic_abc(prefixes: List[str], *, output_dir: str, segment_ids: Optional[List[int]]=None, model_dir: Optional[str]=None, device: Optional[str]=None, max_new_tokens: int=512, temperature: float=0.8, top_p: float=0.95, seed: Optional[int]=None) -> List[Dict[str, object]]:
    from track.melody.infer import generate
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_segment_ids = segment_ids or [0 for _ in prefixes]
    if len(resolved_segment_ids) != len(prefixes):
        raise ValueError('segment_ids must have the same length as prefixes')
    outputs: List[Dict[str, object]] = []
    for index, (prefix, segment_id) in enumerate(zip(prefixes, resolved_segment_ids)):
        abc_text = generate(prefix=prefix, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=True, seed=None if seed is None else seed + index, model_dir=model_dir, device=device)
        abc_path = out_dir / f'segment_{segment_id:04d}_bar_{index:04d}.abc'
        abc_path.write_text(abc_text, encoding='utf-8')
        outputs.append({'index': index, 'segment_id': int(segment_id), 'prefix': prefix, 'abc': abc_text, 'abc_path': str(abc_path)})
    return outputs
__all__ = ['generate_symbolic_abc']

def generate_blueprint_score(blueprint, *, output_dir, model_dir=None, device=None, seed=123, max_new_tokens=512, max_attempts=3, budget=None, policy_checkpoint=''):
    import json
    from track.melody.infer import generate
    from symbolic.events import abc_to_events, constrain_events, export_midi
    from symbolic.notation import decode_mupt
    from planning.timeline import merge_overlapping_segments, TempoMap
    blueprint = merge_overlapping_segments(blueprint)
    tempo = TempoMap.from_blueprint(blueprint)
    total = max((s.end_time for s in blueprint.segments))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records, all_events = ([], [])
    for si, segment in enumerate(blueprint.segments):
        for ti, track in enumerate(segment.tracks):
            prefix = f'X:1<n>T:{track.track_id or track.role}<n>M:{segment.meter}<n>L:1/8<n>Q:1/4={segment.tempo_bpm}<n>% role={track.role}; register={track.register}; density={track.density:.3f}; phrase={track.phrase_function}; section={segment.section}; chords={','.join(segment.chords)}<n>K:{segment.global_key}<n>'
            failures = []
            for attempt in range(max_attempts):
                if budget is not None:
                    budget.consume('mupt')
                abc = generate(prefix, model_dir=model_dir, device=device, max_new_tokens=max_new_tokens, seed=seed + si * 10007 + ti * 101 + attempt, policy_checkpoint=policy_checkpoint)
                raw_path = root / f'segment_{segment.segment_id:04d}_track_{ti:02d}_attempt_{attempt}.abc'
                raw_path.write_text(abc.replace('<n>', '\n'), encoding='utf-8')
                try:
                    normalized, notation = decode_mupt(abc)
                    events = abc_to_events(abc, tempo_map=tempo, start_time=segment.start_time, track_id=track.track_id or track.role, role=track.role, program=track.midi_program)
                    if len({e.track_id for e in events}) > 1:
                        raise ValueError('per-track generation returned multiple voices; refusing to merge them silently')
                    realized = constrain_events(events, track, start=segment.start_time, end=segment.end_time, total_duration=total)
                    if track.density > 0 and (not realized):
                        raise ValueError('no usable notes within requested segment')
                    break
                except Exception as exc:
                    failures.append({'attempt': attempt, 'error': str(exc), 'abc_path': str(raw_path)})
            else:
                (root / 'generation_failure.json').write_text(json.dumps(failures, indent=2))
                raise RuntimeError(f'MuPT produced no valid score for segment {segment.segment_id}, track {ti}: {failures}')
            normalized_path = raw_path.with_suffix('.standard.abc')
            normalized_path.write_text(normalized, encoding='utf-8')
            records.append({'segment_id': segment.segment_id, 'track_id': track.track_id or track.role, 'abc': abc, 'abc_path': str(raw_path), 'prefix': prefix, 'events': [e.to_dict() for e in realized], 'standard_abc_path': str(normalized_path), 'notation': notation, 'policy_checkpoint': policy_checkpoint or None, 'attempts': attempt + 1, 'invalid_attempts': failures, 'postprocess': 'octave_fold_and_density_cap', 'raw_note_count': len(events), 'realized_note_count': len(realized)})
            all_events.extend(realized)
    export_midi(all_events, root / 'score.mid', tempo)
    (root / 'score_events.json').write_text(json.dumps({'events': [e.to_dict() for e in all_events], 'tempo_map': tempo.points, 'duration': total, 'repeat_policy': 'as_written', 'units': 'seconds', 'generation_calls': sum((r['attempts'] for r in records)), 'invalid_outputs': sum((len(r['invalid_attempts']) for r in records))}, indent=2))
    return records
