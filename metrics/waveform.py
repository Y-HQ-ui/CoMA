from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly
SAMPLE_RATE = 32000
DURATION = 15.0
SAMPLES = 480000
VERSION = 'coma_evaluation_mix_32k_15s_v1'

def evaluation_waveform(path):
    waveform, rate = sf.read(path, always_2d=True, dtype='float64')
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError('evaluation requires nonempty finite audio')
    waveform = waveform.mean(axis=1)
    source_samples = len(waveform)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        waveform = resample_poly(waveform, SAMPLE_RATE // divisor, rate // divisor)
    result = np.zeros(SAMPLES, dtype=np.float32)
    count = min(len(waveform), SAMPLES)
    result[:count] = waveform[:count]
    return (result, {'version': VERSION, 'sample_rate': SAMPLE_RATE, 'samples': SAMPLES, 'duration_seconds': DURATION, 'source_sample_rate': rate, 'source_samples': source_samples, 'channel_policy': 'mono_mean', 'alignment': 'first_15s_end_zero_padding', 'gain': 'unchanged', 'local_silence': 'retained'})

def prepare_evaluation_mix(source, destination):
    from experiments.provenance import sha256_file
    destination = Path(destination)
    if Path(source).resolve() == destination.resolve():
        raise ValueError('evaluation mix must not overwrite its source')
    waveform, protocol = evaluation_waveform(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, SAMPLE_RATE, waveform)
    return {**protocol, 'source_sha256': sha256_file(source), 'audio_sha256': sha256_file(destination), 'path': str(destination)}

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = prepare_evaluation_mix(args.audio, args.output)
    Path(args.output + '.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
