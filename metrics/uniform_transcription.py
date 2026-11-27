import argparse
import importlib
import json
import math
from pathlib import Path
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from experiments.provenance import sha256_file, canonical_hash
from .waveform import prepare_evaluation_mix, VERSION as MIX_VERSION
VERSION = 'uniform_audio_transcription_v3'

def record_key(row):
    decoding_seed = row.get('decoding_seed', row.get('seed'))
    if decoding_seed is None:
        raise ValueError('MMP records require a decoding_seed (legacy seed accepted)')
    if 'seed' in row and 'decoding_seed' in row and (row['seed'] != decoding_seed):
        raise ValueError('seed and decoding_seed disagree')
    return (row['sample_id'], row['method'], row.get('training_seed'), decoding_seed)

def evaluate_records(records, protocol, transcribe, evaluate, *, evaluation_dir=None):
    protocol = {**protocol, 'evaluation_mix': MIX_VERSION}
    context = nullcontext(str(evaluation_dir)) if evaluation_dir is not None else TemporaryDirectory(prefix='coma-mmp-')
    with context as directory:
        return _evaluate_records(records, protocol, transcribe, evaluate, Path(directory), evaluation_dir is not None)

def _evaluate_records(records, protocol, transcribe, evaluate, directory, persistent):
    required = {'transcriber_checkpoint', 'evaluator_checkpoint', 'prompt', 'decoding', 'score_definition'}
    if not required <= set(protocol) or any((not protocol[k] for k in required)):
        raise ValueError('explicit transcriber/evaluator/prompt/decoding/score definition required')
    if not records:
        raise ValueError('no audio records')
    keys, output = (set(), [])
    for row in records:
        key = record_key(row)
        if key in keys:
            raise ValueError('duplicate MMP evaluation record')
        keys.add(key)
        audio = Path(row['audio'])
        if 'abc' in row or 'symbolic_outputs' in row:
            raise ValueError('uniform evaluation must transcribe every waveform; clean symbolic bypass is forbidden')
        item = {**row, 'training_seed': key[2], 'decoding_seed': key[3], 'source_audio_sha256': sha256_file(audio)}
        try:
            mix = prepare_evaluation_mix(audio, directory / f'{len(output):06d}.wav')
            item.update(audio_sha256=mix['audio_sha256'], evaluation_mix={k: v for k, v in mix.items() if k != 'path' or persistent})
            transcription = transcribe(audio_path=mix['path'], checkpoint=protocol['transcriber_checkpoint'], decoding=protocol['decoding'])
            abc = transcription['abc']
            if transcription.get('checkpoint') != protocol['transcriber_checkpoint'] or not isinstance(abc, str) or (not abc.strip()):
                raise ValueError('transcriber version/output mismatch')
            result = evaluate(abc=abc, checkpoint=protocol['evaluator_checkpoint'], prompt=protocol['prompt'], decoding=protocol['decoding'], score_definition=protocol['score_definition'])
            score = float(result['score'])
            if result.get('checkpoint') != protocol['evaluator_checkpoint'] or not math.isfinite(score) or (not 0 <= score <= 1):
                raise ValueError('evaluator version/score mismatch; explicit [0,1] score required')
            item.update(status='ok', value=score, transcription=transcription, evaluator=result)
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            item.update(status='failed', error=str(exc))
        output.append(item)
    return {'version': VERSION, 'protocol': protocol, 'protocol_id': canonical_hash(protocol), 'records': output, 'policy': 'all systems transcribed with the same checkpoint/prompt; failures retained; no native score bypass'}

def select_measurement(report, audio_path, *, identity=None):
    with TemporaryDirectory(prefix='coma-mmp-check-') as tmp:
        digest = prepare_evaluation_mix(audio_path, Path(tmp) / 'evaluation.wav')['audio_sha256']
    if report.get('version') != VERSION or report.get('protocol_id') != canonical_hash(report['protocol']):
        raise ValueError('invalid uniform transcription protocol')
    matches = [row for row in report['records'] if row.get('audio_sha256') == digest]
    if identity is not None:
        key = record_key(identity)
        matches = [row for row in matches if record_key(row) == key]
    if len(matches) != 1 or matches[0]['status'] != 'ok':
        raise ValueError('need one successful MMP record bound to this exact waveform; supply record_identity when waveforms repeat')
    return (matches[0]['value'], report['protocol_id'])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--transcriber', default='metrics.mmp_backends:transcribe_mt3', help='module:function')
    p.add_argument('--evaluator', default='metrics.mmp_backends:evaluate_mmp', help='module:function')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    def plugin(name):
        module, function = name.split(':')
        return getattr(importlib.import_module(module), function)
    path = Path(args.manifest).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in records:
        audio = Path(row['audio'])
        row['audio'] = str(audio if audio.is_absolute() else path.parent / audio)
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    protocol.update(transcriber_plugin=args.transcriber, evaluator_plugin=args.evaluator)
    protocol = resolve_protocol_paths(protocol, protocol_path.parent)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = evaluate_records(records, protocol, plugin(args.transcriber), plugin(args.evaluator), evaluation_dir=output.parent / (output.stem + '_audio'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all((row['status'] == 'ok' for row in result['records'])) else 1

def resolve_protocol_paths(protocol, base):
    protocol = {**protocol, 'decoding': dict(protocol['decoding'])}

    def path(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (Path(base) / p).resolve())
    if protocol.get('transcriber_plugin', 'metrics.mmp_backends:transcribe_mt3') == 'metrics.mmp_backends:transcribe_mt3':
        protocol['transcriber_checkpoint'] = path(protocol['transcriber_checkpoint'])
        for key in ('mt3_source_root', 'mt3_source_file', 'mt3_python'):
            if protocol['decoding'].get(key):
                protocol['decoding'][key] = path(protocol['decoding'][key])
    return protocol
if __name__ == '__main__':
    raise SystemExit(main())
