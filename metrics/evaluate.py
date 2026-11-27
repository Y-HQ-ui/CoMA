from __future__ import annotations
import argparse
import json
from pathlib import Path

def resolve_metric_paths(value, base):
    paths = {'embedding_path', 'head_path', 'model_dir', 'source_root', 'reference_audio', 'video', 'video_path', 'generated', 'reference', 'target', 'calibration', 'uniform_transcription_report', 'clap_model_dir', 'imagebind_video_embedding', 'imagebind_audio_embedding', 'imagebind_candidates', 'inception_probabilities', 'fad_generated_embeddings', 'fad_reference_embeddings'}
    if isinstance(value, list):
        return [resolve_metric_paths(item, base) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if isinstance(item, str) and item and (key in paths or (key == 'checkpoint' and value.get('model_id') in {'imagebind_huge', 'vggish'})):
            path = Path(item).expanduser()
            result[key] = str(path if path.is_absolute() else (Path(base) / path).resolve())
        else:
            result[key] = resolve_metric_paths(item, base)
    return result

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blueprint', required=True)
    p.add_argument('--audio-manifest', required=True)
    p.add_argument('--symbolic-outputs', default='')
    p.add_argument('--visual-features', default='')
    p.add_argument('--metric-inputs', default='')
    p.add_argument('--output', required=True)
    p.add_argument('--prompt', default='')
    p.add_argument('--conditioning-reference', default='')
    p.add_argument('--reconstruction-reference', default='')
    args = p.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('evaluation output already exists; choose a new path')

    def read(path, default):
        return json.loads(Path(path).read_text()) if path else default
    from planning import load_blueprint_json
    from .protocol import evaluate_metrics
    from experiments.provenance import sha256_file, source_fingerprint
    spec = resolve_metric_paths(read(args.metric_inputs, {}), Path(args.metric_inputs).resolve().parent) if args.metric_inputs else {}
    audio_manifest = read(args.audio_manifest, {})
    if audio_manifest.get('soundtrack_path'):
        path = Path(audio_manifest['soundtrack_path'])
        if not path.is_absolute():
            audio_manifest['soundtrack_path'] = str((Path(args.audio_manifest).resolve().parent / path).resolve())
    visual = read(args.visual_features, None)
    if visual and 'frame_features' in visual:
        visual = visual['frame_features']
    report = evaluate_metrics(blueprint=load_blueprint_json(args.blueprint), symbolic_outputs=read(args.symbolic_outputs, []), audio_manifest=audio_manifest, prompt=args.prompt, reference_audio_path=args.conditioning_reference, reconstruction_reference_audio_path=args.reconstruction_reference, visual_features=visual, metric_inputs=spec, evaluation_dir=output.parent / (output.stem + '_audio'), imagebind_video_embedding=spec.get('imagebind_video_embedding', ''), imagebind_audio_embedding=spec.get('imagebind_audio_embedding', ''), inception_probabilities=spec.get('inception_probabilities', ''), clap_model_dir=spec.get('clap_model_dir', ''))
    report['input_hashes'] = {name: sha256_file(path) for name, path in vars(args).items() if name in {'blueprint', 'audio_manifest', 'symbolic_outputs', 'visual_features', 'metric_inputs'} and path}
    report['source_sha256'] = source_fingerprint(Path(__file__).resolve().parents[1])['sha256']
    report['scope'] = 'per-clip evaluation; distribution metrics require common-valid-set aggregation'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({name: value['status'] for name, value in report['metrics'].items() if name != 'metric_aliases'}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
