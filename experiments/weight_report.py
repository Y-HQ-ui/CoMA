import argparse
import csv
import json
from pathlib import Path
import numpy as np

def summarize(paths):
    if len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError('duplicate trajectory path cannot count as another training run')
    runs = [json.loads(Path(p).read_text()) for p in paths]
    if not runs or any((not run.get('rows') for run in runs)):
        raise ValueError('nonempty measured coefficient trajectories required')
    protocol = runs[0]['protocol']
    names = list(protocol['active_components'])
    combined = []
    for run_id, run in enumerate(runs):
        if run['protocol'] != protocol:
            raise ValueError('different coefficient/measurement/ablation protocols cannot be pooled')
        steps = [row['step'] for row in run['rows']]
        if steps != sorted(set(steps)):
            raise ValueError('duplicate or unordered training steps')
        for row in run['rows']:
            if set(row['weights']) != set(names):
                raise ValueError('coefficient names changed within a run')
            values = np.array([row['weights'][n] for n in names])
            if not np.isfinite(values).all() or not np.isclose(values.sum(), 1) or (values < protocol['weights']['floor'] - 1e-10).any():
                raise ValueError('coefficient vector violates recorded global simplex')
            combined.append({'run': run_id, 'step': row['step'], **row['weights']})
    per_run = np.array([[np.mean([r[n] for r in combined if r['run'] == i]) for n in names] for i in range(len(runs))])
    summary = {}
    for i, name in enumerate(names):
        values = np.array([row[name] for row in combined])
        summary[name] = {'mean_run_equal': float(per_run[:, i].mean()), 'std_of_run_means': float(per_run[:, i].std(ddof=1)) if len(runs) > 1 else None, 'min_all_recorded_steps': float(values.min()), 'max_all_recorded_steps': float(values.max())}
    assert np.isclose(sum((v['mean_run_equal'] for v in summary.values())), 1)
    return {'protocol': protocol, 'source_paths': [str(p) for p in paths], 'runs': len(runs), 'recorded_steps': len(combined), 'rows': combined, 'summary': summary, 'aggregation': 'equal weight per run; within each run mean over all recorded steps', 'figure_scope': 'same rows as table; run-specific lines; no uncertainty band for a single run'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectories', nargs='+')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    report = summarize(args.trajectories)
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('use a new report directory')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'coefficient_report.json').write_text(json.dumps(report, indent=2))
    with (root / 'coefficients.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['component', *next(iter(report['summary'].values())).keys()])
        writer.writeheader()
        for component, stats in report['summary'].items():
            writer.writerow({'component': component, **stats})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in report['protocol']['active_components']:
        for run in range(report['runs']):
            rows = [row for row in report['rows'] if row['run'] == run]
            ax.plot([row['step'] for row in rows], [row[name] for row in rows], marker='.', label=f'{name}, run {run}')
    ax.set(xlabel='Generator PPO update step', ylabel='Global simplex coefficient', ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(root / 'coefficients.png', dpi=160)
    plt.close(fig)
if __name__ == '__main__':
    main()
