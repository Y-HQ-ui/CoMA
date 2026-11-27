import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .provenance import canonical_hash
COMPARISON_FIELDS = ('split_id', 'sample_ids', 'input_condition', 'seeds', 'max_generation_calls', 'evaluation_version', 'evaluation_settings', 'prompt_source', 'reference_source', 'duration_seconds', 'aggregation')

def _feature(entry, name, dimensions):
    value = entry.get(name)
    if value is None and entry.get(name + '_path'):
        value = np.load(entry[name + '_path'], allow_pickle=False)
    if value is None:
        raise ValueError(f'distribution metrics require {name}; per-clip scalar scores cannot be averaged')
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != dimensions or (not len(value)) or (not np.isfinite(value).all()):
        raise ValueError(f'{name} must be finite [N,{dimensions}]')
    return value

def _distribution_score(entries, metric):
    from metrics.protocol import frechet_embeddings, inception_score
    if metric in {'FAD', 'frechet_audio_distance'}:
        generated = np.concatenate([_feature(e, 'embeddings', 128) for e in entries])
        reference = np.concatenate([_feature(e, 'reference_embeddings', 128) for e in entries])
        return frechet_embeddings(generated, reference)
    clips = []
    for entry in entries:
        patches = _feature(entry, 'probabilities', 10)
        if (patches < 0).any() or not np.allclose(patches.sum(1), 1, atol=1e-05):
            raise ValueError('IS requires normalized ten-genre patch probabilities')
        clips.append(patches.mean(0))
    return inception_score(clips)

def load_run(path):
    path = Path(path).resolve()
    run = json.loads(path.read_text())
    for record in run['records']:
        for entry in record.get('metrics', {}).values():
            for key in ('embeddings_path', 'reference_embeddings_path', 'probabilities_path'):
                if entry.get(key):
                    value = Path(entry[key])
                    entry[key] = str(value if value.is_absolute() else path.parent / value)
    return run

def aggregate_runs(runs, metric_names):
    if not runs:
        raise ValueError('no runs')
    comparison = {k: runs[0]['protocol'].get(k) for k in COMPARISON_FIELDS}
    if any(({k: run['protocol'].get(k) for k in COMPARISON_FIELDS} != comparison for run in runs)):
        raise ValueError('incompatible table protocols; export separate tables')
    samples, seeds = (comparison['sample_ids'], comparison['seeds'])
    if not samples or not seeds or len(set(samples)) != len(samples) or (len(set(seeds)) != len(seeds)):
        raise ValueError('unique nonempty sample_ids and decoding seeds required')
    movie_ids, indexed, rows = ({}, [], [])
    for run in runs:
        if run['method'] in {r['method'] for r in rows}:
            raise ValueError('duplicate method')
        training_seeds = run['protocol'].get('training_seeds', [None])
        if not training_seeds or len(set(training_seeds)) != len(training_seeds):
            raise ValueError('unique training seeds required; frozen baselines use [null]')
        records = {}
        for sid, movie in run['protocol'].get('movie_ids', {}).items():
            if sid in movie_ids and movie_ids[sid] != movie:
                raise ValueError('inconsistent movie identity')
            movie_ids[sid] = movie
        for record in run['records']:
            sid = record['sample_id']
            train_seed = record.get('training_seed', training_seeds[0] if len(training_seeds) == 1 else None)
            seed = record.get('decoding_seed', record.get('seed'))
            key = (sid, train_seed, seed)
            if sid not in samples or train_seed not in training_seeds or seed not in seeds or (key in records):
                raise ValueError('unexpected or duplicate sample/training-seed/decoding-seed record')
            movie = record.get('movie_id', movie_ids.get(sid))
            if not movie or (sid in movie_ids and movie_ids[sid] != movie):
                raise ValueError('consistent movie_id required for every clip')
            movie_ids[sid] = movie
            records[key] = record
        indexed.append((records, training_seeds))
        rows.append({'method': run['method'], 'n_test_clips': len(samples), 'training_seeds': training_seeds, 'decoding_seeds': seeds, 'metrics': {}})
    if any((not movie_ids.get(sid) for sid in samples)):
        raise ValueError('movie_ids required even for clips missing from every run')
    common_sets = {}
    distribution_metrics = {'FAD', 'frechet_audio_distance', 'IS', 'inception_score'}
    for metric in metric_names:
        participating, valid_sets, backends = ([], [], set())
        for index, (run, (records, train_seeds)) in enumerate(zip(runs, indexed)):
            if metric in run.get('not_applicable_metrics', []):
                rows[index]['metrics'][metric] = {'status': 'not_applicable', 'mean': None}
                continue
            participating.append(index)
            valid, failures = (set(), 0)
            for sid in samples:
                complete = True
                for train_seed in train_seeds:
                    for seed in seeds:
                        record = records.get((sid, train_seed, seed), {})
                        entry = record.get('metrics', {}).get(metric, {})
                        ok = record.get('status') == 'ok' and entry.get('status') == 'ok'
                        if ok:
                            if not entry.get('backend'):
                                raise ValueError('successful metric requires evaluator backend')
                            backends.add(entry['backend'])
                            if metric not in distribution_metrics and (not np.isfinite(float(entry['value']))):
                                raise ValueError('nonfinite result')
                        else:
                            failures += 1
                            complete = False
                if complete:
                    valid.add(sid)
            valid_sets.append(valid)
            expected = len(samples) * len(train_seeds) * len(seeds)
            rows[index]['metrics'][metric] = {'failed_records': failures, 'expected_records': expected, 'failure_rate': failures / expected, 'valid_clips': len(valid), 'clip_failure_rate': 1 - len(valid) / len(samples)}
        if len(backends) > 1:
            raise ValueError('methods use incompatible evaluators')
        common = sorted(set.intersection(*valid_sets)) if valid_sets else []
        common_sets[metric] = {'sample_ids': common, 'n_common_clips': len(common), 'participating_methods': [rows[i]['method'] for i in participating]}
        for index in participating:
            records, train_seeds = indexed[index]
            result = rows[index]['metrics'][metric]
            result.update(n_common_clips=len(common), n_movies=len({movie_ids[sid] for sid in common}))
            if not common:
                result.update(status='unavailable', mean=None, reason='empty common valid set')
                continue
            if metric in {'IS', 'inception_score'} and len(common) < 2:
                result.update(status='unavailable', mean=None, reason='IS requires at least two common clips')
                continue

            def entry(sid, train_seed, seed):
                return records[sid, train_seed, seed]['metrics'][metric]
            if metric in distribution_metrics:
                seed_scores = [{'training_seed': t, 'decoding_seed': s, 'value': _distribution_score([entry(sid, t, s) for sid in common], metric)} for t in train_seeds for s in seeds]
                result.update(mean=float(np.mean([x['value'] for x in seed_scores])), per_seed_combination=seed_scores, aggregation='recomputed_on_common_set_per_seed_combination')
            else:
                clip_means = {sid: float(np.mean([np.mean([entry(sid, t, s)['value'] for s in seeds]) for t in train_seeds])) for sid in common}
                movie_means = {movie: float(np.mean([clip_means[sid] for sid in common if movie_ids[sid] == movie])) for movie in sorted({movie_ids[sid] for sid in common})}
                result.update(mean=float(np.mean(list(movie_means.values()))), per_video=clip_means, per_movie=movie_means, aggregation='decode_then_train_then_clip_then_equal_movie')
            result.update(status='ok', backend=next(iter(backends)))
    return {'comparison': comparison, 'rows': rows, 'common_sets': common_sets, 'source_ids': [r.get('protocol_id', canonical_hash(r['protocol'])) for r in runs]}

def normalized_radar(table, directions):
    result = {r['method']: [] for r in table['rows']}
    ranges = {}
    for metric, direction in directions.items():
        if direction not in {'higher', 'lower'}:
            raise ValueError('direction must be higher/lower')
        values = np.asarray([r['metrics'][metric]['mean'] for r in table['rows']], dtype=float)
        valid = np.isfinite(values)
        lo, hi = (values[valid].min(), values[valid].max()) if valid.any() else (None, None)
        scaled = np.full(len(values), np.nan)
        if valid.any():
            scaled[valid] = (values[valid] - lo) / (hi - lo) if hi > lo else 0.5
        if direction == 'lower':
            scaled = 1 - scaled
        ranges[metric] = {'min': float(lo) if lo is not None else None, 'max': float(hi) if hi is not None else None, 'direction': direction}
        for row, value in zip(table['rows'], scaled):
            result[row['method']].append(float(value) if np.isfinite(value) else None)
    return {'values': result, 'ranges': ranges, 'formula': '(x-min)/(max-min), reversed for lower-is-better; tied axis=0.5'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--directions', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--radar', action='store_true')
    args = p.parse_args()
    directions = json.loads(Path(args.directions).read_text())
    table = aggregate_runs([load_run(x) for x in args.runs], list(directions))
    radar = normalized_radar(table, directions)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'table_data.json').write_text(json.dumps({**table, 'normalization': radar}, indent=2))
    with (root / 'table.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', *directions])
        for row in table['rows']:
            writer.writerow([row['method'], *[row['metrics'][m]['mean'] for m in directions]])
    if args.radar:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        angles = np.linspace(0, 2 * np.pi, len(directions), endpoint=False)
        angles = np.r_[angles, angles[0]]
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for method, values in radar['values'].items():
            ax.plot(angles, [*values, values[0]], label=method)
        ax.set_xticks(angles[:-1], directions)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.1, 1))
        fig.savefig(root / 'radar.png', bbox_inches='tight', dpi=160)
        plt.close(fig)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
