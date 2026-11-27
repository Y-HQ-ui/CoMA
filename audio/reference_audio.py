from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-08:
        return np.zeros_like(values)
    return ((values - lo) / (hi - lo)).astype(np.float32)

def _resample(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    if values.size == 1:
        return np.full(length, float(values[0]), dtype=np.float32)
    return np.interp(np.linspace(0.0, 1.0, length), np.linspace(0.0, 1.0, values.size), values).astype(np.float32)

def _estimate_tempo(onset: np.ndarray, sample_rate: float) -> float:
    if onset.size < 4:
        return 0.0
    centered = onset - float(np.mean(onset))
    spectrum = np.correlate(centered, centered, mode='full')[onset.size - 1:]
    min_lag = max(1, int(sample_rate * 60.0 / 180.0))
    max_lag = min(onset.size - 1, int(sample_rate * 60.0 / 50.0))
    if max_lag <= min_lag:
        return 0.0
    lag = min_lag + int(np.argmax(spectrum[min_lag:max_lag]))
    return float(np.clip(60.0 * sample_rate / max(1, lag), 50.0, 180.0))

def extract_reference_audio_features(audio_path: str, *, temporal_hz: float=2.0, max_llm_points: int=64) -> Dict[str, Any]:
    import soundfile as sf
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f'reference audio does not exist: {audio_path}')
    if temporal_hz <= 0:
        raise ValueError('temporal_hz must be positive')
    waveform, sample_rate = sf.read(str(path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError(f'reference audio is empty or invalid: {audio_path}')
    frame_length = max(256, int(round(sample_rate * 0.05)))
    hop_length = max(128, int(round(sample_rate * 0.025)))
    frames = []
    for start in range(0, max(1, waveform.size), hop_length):
        frame = waveform[start:start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        frames.append(frame)
        if start + frame_length >= waveform.size:
            break
    frame_array = np.stack(frames).astype(np.float32)
    window = np.hanning(frame_length).astype(np.float32)
    spectra = np.abs(np.fft.rfft(frame_array * window, axis=1))
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    spectrum_sum = spectra.sum(axis=1) + 1e-08
    rms = np.sqrt(np.mean(frame_array ** 2, axis=1) + 1e-08)
    flux = np.maximum(0.0, np.diff(spectra, axis=0, prepend=spectra[:1])).mean(axis=1)
    centroid = (spectra * frequencies[None, :]).sum(axis=1) / spectrum_sum
    hf_mask = frequencies >= min(4000.0, sample_rate * 0.25)
    hf_ratio = spectra[:, hf_mask].sum(axis=1) / spectrum_sum if np.any(hf_mask) else np.zeros_like(rms)
    low_mask = frequencies <= min(250.0, sample_rate * 0.12)
    low_ratio = spectra[:, low_mask].sum(axis=1) / spectrum_sum if np.any(low_mask) else np.zeros_like(rms)
    frame_hz = sample_rate / hop_length
    target_length = max(2, int(math.ceil(waveform.size / sample_rate * temporal_hz)))
    energy = _resample(_normalize(rms), target_length)
    onset = _resample(_normalize(flux), target_length)
    brightness = _resample(_normalize(centroid), target_length)
    warmth = _resample(_normalize(low_ratio - hf_ratio), target_length)
    warmth = np.clip(0.5 + 0.5 * (warmth - float(np.mean(warmth))), 0.0, 1.0)
    dynamics = _resample(_normalize(np.abs(np.diff(rms, prepend=rms[:1]))), target_length)
    novelty = _resample(_normalize(np.abs(np.diff(spectra, axis=0, prepend=spectra[:1])).mean(axis=1)), target_length)
    tempo = _estimate_tempo(_normalize(flux), frame_hz)

    def summary(values: np.ndarray) -> Dict[str, float]:
        return {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'min': float(np.min(values)), 'max': float(np.max(values))}
    point_count = min(max_llm_points, target_length)
    return {'path': str(path), 'source': 'reference_audio_waveform', 'sample_rate': int(sample_rate), 'duration': float(waveform.size / sample_rate), 'tempo_bpm': tempo, 'summary': {'energy': summary(energy), 'onset': summary(onset), 'brightness': summary(brightness), 'warmth': summary(warmth), 'dynamics': summary(dynamics), 'novelty': summary(novelty)}, 'trajectory_hz': float(temporal_hz), 'trajectories': {'energy': _resample(energy, point_count).tolist(), 'onset': _resample(onset, point_count).tolist(), 'brightness': _resample(brightness, point_count).tolist(), 'warmth': _resample(warmth, point_count).tolist(), 'dynamics': _resample(dynamics, point_count).tolist(), 'novelty': _resample(novelty, point_count).tolist()}, 'waveform': waveform}

def reference_audio_for_llm(features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not features:
        return {}
    return {key: value for key, value in features.items() if key not in {'waveform'}}
__all__ = ['extract_reference_audio_features', 'reference_audio_for_llm']
