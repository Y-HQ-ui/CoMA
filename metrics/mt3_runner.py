import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

def check(source_root, checkpoint):
    source_root, checkpoint = (Path(source_root).resolve(), Path(checkpoint).resolve())
    required = ['mt3/gin/model.gin', 'mt3/gin/mt3.gin', 'mt3/network.py', 'mt3/models.py']
    missing = ['source:' + p for p in required if not (source_root / p).is_file()]
    if not checkpoint.exists():
        missing.append('checkpoint:' + str(checkpoint))
    for package in ['tensorflow', 'jax', 'gin', 'seqio', 't5', 't5x', 'note_seq', 'soundfile', 'scipy']:
        try:
            present = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append('package:' + package)
    return {'status': 'ready_for_load' if not missing else 'unavailable', 'missing': missing}

def transcribe(audio, checkpoint, source_root, midi):
    source_root, checkpoint = (Path(source_root).resolve(), Path(checkpoint).resolve())
    midi = Path(midi).resolve()
    if midi.exists():
        raise FileExistsError(midi)
    report = check(source_root, checkpoint)
    if report['missing']:
        raise RuntimeError(json.dumps(report))
    sys.path.insert(0, str(source_root))
    import mt3
    if not Path(mt3.__file__).resolve().is_relative_to(source_root):
        raise ValueError('imported MT3 package differs from the selected local checkout')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import tensorflow.compat.v2 as tf
    tf.config.set_visible_devices([], 'GPU')
    import gin
    gin.add_config_file_search_path(str(source_root))
    if __package__:
        from .mt3_official import InferenceModel
    else:
        from mt3_official import InferenceModel
    import note_seq
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    waveform, rate = sf.read(audio, always_2d=True, dtype='float32')
    waveform = waveform.mean(1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError('finite nonempty audio required')
    if rate != 16000:
        divisor = gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
    model = InferenceModel(str(checkpoint), str(source_root))
    sequence = model(waveform.astype(np.float32))
    if not sequence.notes:
        raise ValueError('MT3 returned no notes')
    midi.parent.mkdir(parents=True, exist_ok=True)
    note_seq.sequence_proto_to_midi_file(sequence, str(midi))
    return {'model': 'mt3', 'notes': len(sequence.notes), 'sample_rate': 16000, 'midi': str(midi)}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--midi')
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args(argv)
    if args.preflight:
        report = check(args.source_root, args.checkpoint)
        print(json.dumps(report, indent=2))
        return int(report['status'] != 'ready_for_load')
    if not args.audio or not args.midi:
        p.error('--audio and --midi are required for inference')
    print(json.dumps(transcribe(args.audio, args.checkpoint, args.source_root, args.midi)))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
