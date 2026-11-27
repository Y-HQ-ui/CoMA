from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from audio.reference_audio import extract_reference_audio_features
from planning.blueprint_schema import MusicBlueprint
from reward.evidence import extract_audio_features, extract_symbolic_features

def _ok(value: float, backend: str, **details: Any) -> Dict[str, Any]:
    return {'status': 'ok', 'value': float(value), 'backend': backend, **details}

def _missing(reason: str, backend: str='not_run') -> Dict[str, Any]:
    return {'status': 'unavailable', 'value': None, 'backend': backend, 'reason': reason}

def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

def _safe_corr(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float32).reshape(-1)
    right_array = np.asarray(right, dtype=np.float32).reshape(-1)
    length = min(left_array.size, right_array.size)
    if length < 2:
        return 0.0
    left_array = left_array[:length]
    right_array = right_array[:length]
    if float(np.ptp(left_array)) < 1e-08 or float(np.ptp(right_array)) < 1e-08:
        return 1.0 - min(1.0, abs(float(np.mean(left_array)) - float(np.mean(right_array))))
    return _clip01(0.5 * (float(np.corrcoef(left_array, right_array)[0, 1]) + 1.0))

def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    length = min(left.size, right.size)
    if length == 0:
        return 0.0
    left = left[:length]
    right = right[:length]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))

def _load_audio(path: str) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    waveform, sample_rate = sf.read(path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        raise ValueError(f'empty audio: {path}')
    return (waveform, int(sample_rate))

def _resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32)
    from scipy.signal import resample_poly
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(waveform, target_rate // divisor, source_rate // divisor).astype(np.float32)

def _audio_embedding(path: str) -> np.ndarray:
    waveform, sample_rate = _load_audio(path)
    waveform = _resample_waveform(waveform, sample_rate, 16000)
    frame_length = 1024
    hop_length = 512
    frames = []
    for start in range(0, max(1, waveform.size), hop_length):
        frame = waveform[start:start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        frames.append(frame)
        if start + frame_length >= waveform.size:
            break
    frame_array = np.stack(frames)
    spectra = np.abs(np.fft.rfft(frame_array * np.hanning(frame_length), axis=1)) + 1e-08
    log_spectra = np.log(spectra)
    band_edges = np.linspace(0, log_spectra.shape[1], 25, dtype=np.int32)
    bands = np.stack([log_spectra[:, band_edges[index]:max(band_edges[index] + 1, band_edges[index + 1])].mean(axis=1) for index in range(24)], axis=1)
    rms = np.sqrt(np.mean(frame_array ** 2, axis=1) + 1e-08)
    zero_cross = np.mean(np.abs(np.diff(np.signbit(frame_array), axis=1)), axis=1)
    features = np.concatenate([bands.mean(axis=0), bands.std(axis=0), np.quantile(rms, [0.1, 0.25, 0.5, 0.75, 0.9]), np.quantile(zero_cross, [0.1, 0.5, 0.9])])
    return features.astype(np.float64)

def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, 0.0, None)
    return vectors * np.sqrt(values)[None, :] @ vectors.T

def _frechet_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.atleast_2d(np.asarray(left, dtype=np.float64))
    right = np.atleast_2d(np.asarray(right, dtype=np.float64))
    left_mean = left.mean(axis=0)
    right_mean = right.mean(axis=0)
    left_cov = np.cov(left, rowvar=False) if left.shape[0] > 1 else np.eye(left.shape[1]) * 1e-06
    right_cov = np.cov(right, rowvar=False) if right.shape[0] > 1 else np.eye(right.shape[1]) * 1e-06
    covariance_product = _matrix_sqrt_psd(_matrix_sqrt_psd(left_cov) @ right_cov @ _matrix_sqrt_psd(left_cov))
    delta = left_mean - right_mean
    return float(delta @ delta + np.trace(left_cov + right_cov - 2.0 * covariance_product))

def _audio_paths(audio_manifest: Optional[Dict[str, Any]]) -> List[str]:
    if not audio_manifest:
        return []
    paths = [str(path) for path in audio_manifest.get('segment_paths', []) if Path(path).exists()]
    soundtrack = str(audio_manifest.get('soundtrack_path', ''))
    if not paths and soundtrack and Path(soundtrack).exists():
        paths.append(soundtrack)
    return paths

def _symbolic_text_by_segment(symbolic_outputs: Optional[List[Dict[str, Any]]]) -> Dict[int, str]:
    grouped: Dict[int, List[str]] = {}
    for item in symbolic_outputs or []:
        grouped.setdefault(int(item.get('segment_id', 0)), []).append(str(item.get('abc', '')))
    return {segment_id: '\n'.join(items) for segment_id, items in grouped.items()}

def _rhythm_emotion_color_metrics(blueprint: MusicBlueprint, audio_paths: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not audio_paths:
        unavailable = _missing('generated segment audio is required', 'waveform')
        return (unavailable, unavailable, unavailable)
    visual_rhythm = []
    visual_arousal = []
    visual_valence = []
    visual_brightness = []
    visual_warmth = []
    audio_rhythm = []
    audio_arousal = []
    audio_valence = []
    audio_brightness = []
    audio_warmth = []
    for index, segment in enumerate(blueprint.segments[:len(audio_paths)]):
        prior = segment.video_prior_5d
        features = extract_audio_features(audio_paths[index])
        visual_rhythm.append(0.65 * prior.get('motion_intensity', 0.5) + 0.35 * prior.get('cut_density', 0.5))
        visual_arousal.append(0.7 * prior.get('motion_intensity', 0.5) + 0.3 * prior.get('visual_novelty', 0.5))
        visual_valence.append(0.6 * prior.get('brightness', 0.5) + 0.4 * prior.get('color_warmth', 0.5))
        visual_brightness.append(prior.get('brightness', 0.5))
        visual_warmth.append(prior.get('color_warmth', 0.5))
        audio_rhythm.append(float(np.mean(np.asarray(features['onset'], dtype=np.float32))))
        audio_arousal.append(float(np.mean(np.asarray(features['rms'], dtype=np.float32))))
        centroid = float(np.mean(np.asarray(features['spectral_centroid'], dtype=np.float32)))
        hf_ratio = float(np.mean(np.asarray(features['hf_ratio'], dtype=np.float32)))
        audio_brightness.append(centroid)
        audio_warmth.append(1.0 - hf_ratio)
        audio_valence.append(_clip01(0.55 * centroid + 0.45 * (1.0 - hf_ratio)))
    rhythm = _safe_corr(visual_rhythm, audio_rhythm)
    emotion = 0.5 * _safe_corr(visual_arousal, audio_arousal) + 0.5 * _safe_corr(visual_valence, audio_valence)
    color = 0.5 * _safe_corr(visual_brightness, audio_brightness) + 0.5 * _safe_corr(visual_warmth, audio_warmth)
    return (_ok(rhythm, 'segment_audio_visual_correlation'), _ok(emotion, 'valence_arousal_correlation'), _ok(color, 'brightness_warmth_correlation'))

def _chroma_diversity(audio_path: str) -> float:
    waveform, sample_rate = _load_audio(audio_path)
    waveform = _resample_waveform(waveform, sample_rate, 22050)
    frame_length = 4096
    hop_length = 2048
    chroma_rows = []
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / 22050)
    valid = frequencies > 30.0
    pitch_classes = np.mod(np.rint(69.0 + 12.0 * np.log2(np.maximum(frequencies, 1e-08) / 440.0)), 12).astype(int)
    for start in range(0, max(1, waveform.size), hop_length):
        frame = waveform[start:start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        magnitude = np.abs(np.fft.rfft(frame * np.hanning(frame_length)))
        chroma = np.zeros(12, dtype=np.float64)
        for pitch_class in range(12):
            mask = valid & (pitch_classes == pitch_class)
            chroma[pitch_class] = float(np.sum(magnitude[mask]))
        chroma /= chroma.sum() + 1e-08
        chroma_rows.append(chroma)
        if start + frame_length >= waveform.size:
            break
    chroma_matrix = np.stack(chroma_rows)
    entropy = -np.sum(chroma_matrix * np.log(chroma_matrix + 1e-08), axis=1) / math.log(12.0)
    temporal_change = np.mean(np.linalg.norm(np.diff(chroma_matrix, axis=0), axis=1)) if chroma_matrix.shape[0] > 1 else 0.0
    return _clip01(0.65 * float(np.mean(entropy)) + 0.35 * min(1.0, float(temporal_change)))

def _multi_agent_music_proficiency(blueprint: MusicBlueprint, symbolic_outputs: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    symbolic_by_segment = _symbolic_text_by_segment(symbolic_outputs)
    scores = []
    for segment in blueprint.segments:
        abc_text = symbolic_by_segment.get(segment.segment_id)
        source = 'abc' if abc_text else 'blueprint_prefix'
        if not abc_text:
            from planning.blueprint_to_abc import blueprint_to_mupt_prefixes
            one_segment = MusicBlueprint(blueprint.global_plan, [segment], blueprint.source)
            abc_text = '\n'.join(blueprint_to_mupt_prefixes(one_segment))
        features = extract_symbolic_features(abc_text, duration=max(0.1, segment.end_time - segment.start_time), key=segment.global_key)
        pitches = np.asarray(features.get('pitches', []), dtype=np.float32)
        durations = np.asarray(features.get('durations', []), dtype=np.float32)
        pitch_variety = min(1.0, len(np.unique(np.mod(pitches, 12))) / 8.0) if pitches.size else 0.0
        rhythmic_variety = min(1.0, len(np.unique(np.round(durations, 3))) / 4.0) if durations.size else 0.0
        interval_score = 1.0
        if pitches.size > 1:
            interval_score = 1.0 - min(1.0, float(np.mean(np.maximum(0.0, np.abs(np.diff(pitches)) - 12.0))) / 12.0)
        density_target = 1.5 + 8.0 * (0.55 * segment.video_prior_5d.get('motion_intensity', 0.5) + 0.45 * segment.video_prior_5d.get('cut_density', 0.5))
        density_score = 1.0 - min(1.0, abs(float(features.get('note_density', 0.0)) - density_target) / max(1.0, density_target))
        track_score = min(1.0, len(segment.tracks) / 4.0)
        scores.append(0.25 * pitch_variety + 0.2 * rhythmic_variety + 0.2 * interval_score + 0.2 * density_score + 0.15 * track_score)
    if not scores:
        return _missing('symbolic plan is required', 'symbolic_structure')
    return _ok(float(np.mean(scores)), 'symbolic_structure_composite')

def _dynamic_variation(audio_path: str) -> float:
    features = extract_audio_features(audio_path)
    rms = np.asarray(features['rms'], dtype=np.float32)
    if rms.size < 2:
        return 0.0
    return _clip01(0.55 * float(np.std(rms)) + 0.45 * float(np.mean(np.abs(np.diff(rms)))))

def _si_sdr(generated_path: str, reference_path: str) -> float:
    generated, generated_rate = _load_audio(generated_path)
    reference, reference_rate = _load_audio(reference_path)
    target_rate = min(generated_rate, reference_rate)
    generated = _resample_waveform(generated, generated_rate, target_rate)
    reference = _resample_waveform(reference, reference_rate, target_rate)
    length = min(generated.size, reference.size)
    generated = generated[:length] - float(np.mean(generated[:length]))
    reference = reference[:length] - float(np.mean(reference[:length]))
    scale = float(np.dot(generated, reference) / (np.dot(reference, reference) + 1e-12))
    target = scale * reference
    noise = generated - target
    return float(10.0 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))

def _inception_score(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError('inception probabilities must have shape [N, C]')
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    marginal = probabilities.mean(axis=0, keepdims=True)
    return float(np.exp(np.mean(np.sum(probabilities * (np.log(probabilities) - np.log(marginal)), axis=1))))

def _clap_similarity(prompt: str, audio_path: str, model_dir: str, device: Optional[str]) -> float:
    from .frozen_backends import clap_embeddings
    from .protocol import imagebind_score
    values, text, _ = clap_embeddings(audio_path, {'model_id': 'laion/larger_clap_music_and_speech', 'model_dir': model_dir, 'device': device or 'cpu', 'segment_seconds': 10.0}, prompt)
    return _clip01((imagebind_score(text, values.mean(0)) + 1) / 2)
