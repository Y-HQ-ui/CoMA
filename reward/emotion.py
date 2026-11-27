from functools import lru_cache
from math import gcd
import os
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from scipy.stats import spearmanr
from experiments.provenance import sha256_file

def aligned_correlation(times, predictions, target):
    times, predictions = (np.asarray(times, float), np.asarray(predictions, float))
    source_times = np.asarray(target['times'], float)
    values = np.asarray(target['values'], float)
    if target.get('used_target_audio') is not False or not target.get('source'):
        raise ValueError('emotion_target requires silent-video source and used_target_audio=false')
    if times.ndim != 1 or predictions.shape != (len(times), 2) or source_times.ndim != 1 or (values.shape != (len(source_times), 2)) or (len(source_times) < 2) or (np.diff(source_times) <= 0).any() or (not all((np.isfinite(x).all() for x in (times, predictions, source_times, values)))):
        raise ValueError('VA arrays must be finite, ordered times with [valence, arousal] pairs')
    mask = (times >= source_times[0]) & (times <= source_times[-1])
    if mask.sum() < 3:
        raise ValueError('fewer than three common VA windows')
    aligned = np.column_stack([np.interp(times[mask], source_times, values[:, i]) for i in range(2)])
    scores = []
    for i in range(2):
        if np.ptp(aligned[:, i]) == 0 or np.ptp(predictions[mask, i]) == 0:
            raise ValueError('constant VA trajectory has undefined Spearman correlation')
        scores.append(float(spearmanr(aligned[:, i], predictions[mask, i]).statistic))
    return (float(np.mean(scores)), int(mask.sum()))

@lru_cache(maxsize=2)
def _models(embedding_path, head_path, embedding_sha256, head_sha256):
    try:
        from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
    except ImportError as exc:
        raise RuntimeError('emotion reward requires Essentia with TensorFlow support and local MusicNN graphs') from exc
    embedding = TensorflowPredictMusiCNN(graphFilename=embedding_path, output='model/dense/BiasAdd')
    head = TensorflowPredict2D(graphFilename=head_path, output='model/Identity')
    return (embedding, head)

@lru_cache(maxsize=4)
def _fingerprint(path, size, mtime_ns):
    return sha256_file(path)

def preload_musicnn(*, embedding_path=None, head_path=None):
    paths = [embedding_path or os.getenv('COMA_MUSICNN_EMBEDDING', ''), head_path or os.getenv('COMA_EMOTION_HEAD', '')]
    if any((not p or not Path(p).is_file() for p in paths)):
        raise FileNotFoundError('set COMA_MUSICNN_EMBEDDING and COMA_EMOTION_HEAD to local .pb graphs')
    paths = [str(Path(p).resolve()) for p in paths]
    hashes = [_fingerprint(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths]
    _models(*paths, *hashes)

def evaluate_musicnn(*, audio_path, evidence, embedding_path=None, head_path=None):
    import soundfile as sf
    paths = [embedding_path or os.getenv('COMA_MUSICNN_EMBEDDING', ''), head_path or os.getenv('COMA_EMOTION_HEAD', '')]
    for path in paths:
        if not path or not Path(path).is_file():
            raise FileNotFoundError('set COMA_MUSICNN_EMBEDDING and COMA_EMOTION_HEAD to local .pb graphs')
    paths = [str(Path(p).resolve()) for p in paths]
    target = evidence.get('emotion_target')
    if not target:
        raise ValueError('six-component reward requires emotion_target from silent-video annotation')
    provenance = {'checkpoint': paths[1], 'model': 'emomusic-msd-musicnn-1', 'checkpoint_sha256': {p: _fingerprint(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths}, 'audio_sha256': sha256_file(audio_path), 'window_seconds': 3.0, 'hop_seconds': 0.5, 'sample_rate': 16000, 'axis_order': ['valence', 'arousal'], 'lag_seconds': 0.0}
    wave, sr = sf.read(audio_path, always_2d=True)
    wave = wave.mean(1)
    if sr != 16000:
        divisor = gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // divisor, sr // divisor)
    if not np.isfinite(wave).all():
        raise ValueError('nonfinite audio for emotion reward')
    starts = range(0, len(wave) - 48000 + 1, 8000)
    if len(starts) < 3:
        return {**provenance, 'status': 'unavailable', 'reason': 'fewer than three complete 3 s VA windows'}
    embedding, head = _models(*paths, *[provenance['checkpoint_sha256'][p] for p in paths])
    predictions, times = ([], [])
    for start in starts:
        prediction = np.asarray(head(embedding(wave[start:start + 48000].astype(np.float32))), float)
        if prediction.size == 0 or prediction.shape[-1] != 2 or (not np.isfinite(prediction).all()):
            raise ValueError('emotion graph must output finite [valence, arousal] pairs')
        predictions.append(prediction.reshape(-1, 2).mean(0).tolist())
        times.append((start + 24000) / 16000)
    provenance.update(times=times, values=predictions)
    try:
        score, count = aligned_correlation(times, predictions, target)
    except ValueError as exc:
        if not str(exc).startswith(('fewer than', 'constant VA')):
            raise
        return {**provenance, 'status': 'unavailable', 'reason': str(exc)}
    return {**provenance, 'status': 'ok', 'correlation': score, 'common_windows': count}
