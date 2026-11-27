from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from experiments.provenance import sha256_file
from experiments.statistics import crossed_human_bootstrap
from experiments.playback import prepare_audio, prepare_video

def design(spec_path, output_dir):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('use a new study directory')
    participants = spec['participants']
    videos = spec['videos']
    dimensions = spec['dimensions']
    if not participants or len(set(participants)) != len(participants) or (not videos) or (not dimensions) or (len(set(dimensions)) != len(dimensions)):
        raise ValueError('unique participants/dimensions and nonempty videos required')
    count = int(spec['videos_per_participant'])
    if not 1 <= count <= len(videos):
        raise ValueError('invalid number of videos per participant')
    systems = list(videos[0]['audio'])
    if len(systems) < 2 or len({v['video_id'] for v in videos}) != len(videos):
        raise ValueError('at least two systems and unique videos required')
    lower, upper = spec.get('rating_range', [1, 5])
    if not lower < upper:
        raise ValueError('invalid rating scale')
    resolved = {}
    for video in videos:
        if set(video['audio']) != set(systems):
            raise ValueError('every selected video must have every system: paired allocation')
        for kind, value in [('video', video['video']), *video['audio'].items()]:
            source = Path(value)
            source = source if source.is_absolute() else spec_path.parent / source
            if not source.is_file():
                raise FileNotFoundError(source)
            resolved[video['video_id'], kind] = source.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'stimuli').mkdir()
    seed = spec.get('seed', 2026)
    rng = random.Random(seed)
    key, assets = ([], {})
    for identity, source in resolved.items():
        opaque = hashlib.sha256((str(seed) + '/' + '/'.join(identity)).encode()).hexdigest()[:20]
        is_video = identity[1] == 'video'
        destination = root / 'stimuli' / (opaque + (source.suffix.lower() if is_video else '.wav'))
        processing = (prepare_video if is_video else prepare_audio)(source, destination)
        assets[identity] = {'path': str(destination.relative_to(root)), 'sha256': sha256_file(destination), 'processing': processing}
    coverage = {video['video_id']: 0 for video in videos}
    for pi, participant in enumerate(participants):
        candidates = list(videos)
        rng.shuffle(candidates)
        candidates.sort(key=lambda v: coverage[v['video_id']])
        selected = candidates[:count]
        trials = []
        for video in selected:
            vid = video['video_id']
            coverage[vid] += 1
            for method in systems:
                stimulus = f'p{pi:04d}_s{len(trials):04d}'
                trials.append({'stimulus_id': stimulus, 'video': assets[vid, 'video']['path'], 'audio': assets[vid, method]['path'], 'ratings': {dimension: None for dimension in dimensions}})
                key.append({'participant_id': participant, 'stimulus_id': stimulus, 'video_id': vid, 'method': method, 'audio_sha256': assets[vid, method]['sha256']})
        rng.shuffle(trials)
        for order, trial in enumerate(trials):
            trial['presentation_order'] = order
        form = {'participant_id': participant, 'instructions': spec['instructions'], 'dimensions': dimensions, 'rating_range': [lower, upper], 'trials': trials}
        (root / f'participant_{pi:04d}.json').write_text(json.dumps(form, ensure_ascii=False, indent=2))
    private = {'study_id': spec['study_id'], 'seed': seed, 'dimensions': dimensions, 'stimulus_processing': [{'video_id': vid, 'kind': kind, **asset} for (vid, kind), asset in assets.items()], 'rating_range': [lower, upper], 'key': key, 'allocation': 'each participant rates every system on each assigned video', 'participants': len(participants), 'unique_video_ids_available': len(videos), 'unique_video_ids_assigned': sum((v > 0 for v in coverage.values())), 'video_coverage': coverage, 'independence_assumption': 'video IDs are the declared statistical units; related clips require movie-level clustering', 'expected_audiovisual_trials': len(key), 'expected_dimension_ratings': len(key) * len(dimensions), 'spec_sha256': sha256_file(spec_path), 'historical_study': False}
    (root / 'private_allocation_key.json').write_text(json.dumps(private, ensure_ascii=False, indent=2))
    return private

def analyze(key_path, completed_forms, method_a, method_b, *, replicates=5000, seed=123):
    key = json.loads(Path(key_path).read_text())
    lookup = {(r['participant_id'], r['stimulus_id']): r for r in key['key']}
    rows, seen = ([], set())
    for filename in completed_forms:
        form = json.loads(Path(filename).read_text())
        participant = form['participant_id']
        for trial in form['trials']:
            pair = (participant, trial['stimulus_id'])
            if pair not in lookup or pair in seen:
                raise ValueError('unknown/duplicate participant stimulus')
            seen.add(pair)
            truth = lookup[pair]
            if set(trial['ratings']) != set(key['dimensions']):
                raise ValueError('rating dimensions differ from allocation')
            for dimension, value in trial['ratings'].items():
                if value is None or isinstance(value, bool) or (not key['rating_range'][0] <= value <= key['rating_range'][1]):
                    raise ValueError('missing/out-of-range rating; do not silently drop incomplete cells')
                rows.append({**truth, 'dimension': dimension, 'rating': float(value)})
    if seen != set(lookup):
        raise ValueError('incomplete assigned study; report missingness explicitly before a separate incomplete-data analysis')
    if not {method_a, method_b} <= {r['method'] for r in rows}:
        raise ValueError('comparison methods absent')
    statistics = {dimension: crossed_human_bootstrap([r for r in rows if r['dimension'] == dimension], method_a, method_b, replicates=replicates, seed=seed) for dimension in key['dimensions']}
    return {'study_id': key['study_id'], 'comparison': [method_a, method_b], 'ratings': rows, 'statistics': statistics, 'counted_audiovisual_trials': len(seen), 'counted_dimension_ratings': len(rows), 'inference': 'paired differences and two-way clustered confidence intervals; no uncomputed significance claims'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    create = sub.add_parser('design')
    create.add_argument('--spec', required=True)
    create.add_argument('--output-dir', required=True)
    evaluate = sub.add_parser('analyze')
    evaluate.add_argument('--key', required=True)
    evaluate.add_argument('--forms', nargs='+', required=True)
    evaluate.add_argument('--method-a', required=True)
    evaluate.add_argument('--method-b', required=True)
    evaluate.add_argument('--output', required=True)
    args = p.parse_args()
    if args.action == 'design':
        result = design(args.spec, args.output_dir)
        print(json.dumps({k: result[k] for k in ('expected_audiovisual_trials', 'expected_dimension_ratings')}))
    else:
        result = analyze(args.key, args.forms, args.method_a, args.method_b)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
