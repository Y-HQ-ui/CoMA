import argparse
import importlib.util
import json
from pathlib import Path
import shutil

def check(spec):
    reports = {}
    requirements = {'emotion': (['essentia'], ['embedding_path', 'head_path']), 'genre': (['essentia'], ['embedding_path', 'head_path']), 'clap': (['transformers', 'torch'], ['model_dir']), 'vggish': (['tensorflow', 'tf_slim'], ['source_root', 'checkpoint', 'reference_audio']), 'imagebind': (['torch', 'torchvision', 'torchaudio', 'pytorchvideo', 'decord', 'timm', 'ftfy', 'regex', 'einops'], ['source_root', 'checkpoint'])}
    for name, options in spec.get('backends', {}).items():
        missing = []
        if name not in requirements:
            reports[name] = {'status': 'unavailable', 'missing': ['unknown backend']}
            continue
        packages, paths = requirements[name]
        for package in packages:
            try:
                present = importlib.util.find_spec(package) is not None
            except (ImportError, ValueError):
                present = False
            if not present:
                missing.append('package:' + package)
        for key in paths:
            if not options.get(key) or not Path(options[key]).expanduser().exists():
                missing.append('local_path:' + key)
        if name == 'imagebind' and (not shutil.which('ffmpeg')):
            missing.append('executable:ffmpeg')
        if name == 'emotion' and sorted(options.get('output_axes', [])) != ['arousal', 'valence']:
            missing.append('graph_output_axes')
        if name == 'genre' and (not options.get('logits_output')):
            missing.append('graph_logits_output')
        reports[name] = {'status': 'ready_for_load' if not missing else 'unavailable', 'missing': missing}
    return {'backends': reports, 'verification': 'dependency/file presence only; graph tensors and real weights require execution'}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output')
    args = p.parse_args(argv)
    from .evaluate import resolve_metric_paths
    source = Path(args.config).resolve()
    report = check(resolve_metric_paths(json.loads(source.read_text()), source.parent))
    content = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
    print(content)
    return int(any((row['status'] != 'ready_for_load' for row in report['backends'].values())))
if __name__ == '__main__':
    raise SystemExit(main())
