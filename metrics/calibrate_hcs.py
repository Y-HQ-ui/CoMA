import argparse
import json
from pathlib import Path
import numpy as np
from experiments.provenance import canonical_hash, sha256_file
from .waveform import evaluation_waveform
from .definitions import waveform_features
VERSION = 'coma_hcs_training_p95_v1'

def calibrate(records):
    import librosa
    if not records or any((row.get('split') != 'train' for row in records)):
        raise ValueError('HCS calibration requires only split=train reference audio')
    if len({row['sample_id'] for row in records}) != len(records):
        raise ValueError('duplicate training reference sample_id')
    distances, sources = ([], [])
    for row in records:
        wave, mix = evaluation_waveform(row['audio'])
        features = waveform_features(wave, mix['sample_rate'])
        tc = librosa.feature.tonnetz(chroma=features['chroma'], sr=mix['sample_rate']).T
        valid = features['active'][:-1] & features['active'][1:]
        delta = np.linalg.norm(np.diff(tc, axis=0), axis=1)[valid]
        if len(delta):
            distances.append(delta)
        sources.append({'sample_id': row['sample_id'], 'split': 'train', 'audio_sha256': sha256_file(row['audio']), 'valid_adjacent_pairs': len(delta)})
    if not distances:
        raise ValueError('no active adjacent tonal-centroid pairs in training references')
    values = np.concatenate(distances)
    sigma = float(np.percentile(values, 95, method='linear'))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('training P95 must be finite and positive')
    result = {'version': VERSION, 'sigma_tc': sigma, 'percentile': 95, 'percentile_method': 'linear', 'scale_source': 'training_reference_audio', 'sources': sources, 'adjacent_pairs': len(values), 'extraction': features['protocol'], 'evaluation_mix': mix['version'], 'silence_policy': 'only originally adjacent active pairs; never bridge silent intervals'}
    result['calibration_sha256'] = canonical_hash(result)
    return result

def load_calibration(path):
    record = json.loads(Path(path).read_text())
    digest = record.pop('calibration_sha256')
    if record.get('version') != VERSION or canonical_hash(record) != digest:
        raise ValueError('invalid HCS calibration record')
    if record.get('scale_source') != 'training_reference_audio' or not record.get('sources') or any((row.get('split') != 'train' for row in record['sources'])) or (not np.isfinite(record['sigma_tc'])) or (record['sigma_tc'] <= 0):
        raise ValueError('HCS scale must be frozen from training references')
    return {**record, 'calibration_sha256': digest, 'calibration_file_sha256': sha256_file(path)}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    manifest = Path(args.manifest).resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for row in records:
        audio = Path(row['audio'])
        row['audio'] = str(audio if audio.is_absolute() else manifest.parent / audio)
    result = calibrate(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'sigma_tc': result['sigma_tc'], 'references': len(records)}))
if __name__ == '__main__':
    main()
