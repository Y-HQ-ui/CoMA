from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
from metrics.definitions import correlation
from metrics.protocol import lagged_ncc, onset_f1
VERSION = 'coma_objectives_20261126'
SYMBOLIC_NAMES = ('beat', 'phrase', 'key', 'diversity', 'playability')
AUDIO_NAMES = ('motion', 'cut', 'tonal', 'emotion', 'novelty', 'consistency')
AUDIO_INITIAL_WEIGHTS = dict(zip(AUDIO_NAMES, (0.2, 0.1, 0.15, 0.15, 0.1, 0.3)))

@dataclass(frozen=True)
class RewardProtocol:
    analysis_hz: float = 100.0
    summary_hz: float = 2.0
    max_lag_seconds: float = 0.1
    boundary_tolerance_seconds: float = 0.1
    phrase_silence_seconds: float = 0.25
    onset_width_seconds: float = 0.03
    onset_coefficient: float = 0.5
    envelope_coefficient: float = 0.5
    key_temperature: float = 0.1
    duration_quantum_seconds: float = 0.025
    max_note_duration_seconds: float = 30.0
    min_pitch: int = 21
    max_pitch: int = 108
    max_polyphony_per_track: int = 10

    def __post_init__(self):
        for name in ('analysis_hz', 'summary_hz', 'onset_width_seconds', 'key_temperature', 'duration_quantum_seconds', 'max_note_duration_seconds', 'phrase_silence_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'positive finite {name} required')
        if self.summary_hz > self.analysis_hz or self.max_polyphony_per_track < 1:
            raise ValueError('invalid resolution/polyphony')
        if not 0 <= self.min_pitch <= self.max_pitch <= 127:
            raise ValueError('invalid MIDI pitch range')
        if any((not math.isfinite(x) or x < 0 for x in (self.max_lag_seconds, self.boundary_tolerance_seconds, self.onset_coefficient, self.envelope_coefficient))):
            raise ValueError('invalid lag/tolerance/consistency coefficients')
        if not np.isclose(self.onset_coefficient + self.envelope_coefficient, 1):
            raise ValueError('consistency coefficients must sum to one')

def component(raw, bounded, equation, **evidence):
    if not np.isfinite([raw, bounded]).all() or not 0 <= bounded <= 1:
        raise ValueError('invalid reward component')
    return {'status': 'ok', 'raw': float(raw), 'reward': float(bounded), 'equation': equation, 'evidence': evidence}

def corr_component(function, equation, **evidence):
    try:
        raw = function()
    except ValueError as exc:
        return {'status': 'unavailable', 'raw': None, 'reward': None, 'equation': equation, 'reason': str(exc), 'evidence': evidence}
    return component(raw, (raw + 1) / 2, equation, **evidence)

def entropy(probabilities):
    values = np.asarray(probabilities, float)
    if values.ndim != 1 or values.size < 1 or (not np.isfinite(values).all()) or (values < 0).any() or (values.sum() <= 0):
        raise ValueError('invalid probability distribution')
    values = values / values.sum()
    return float(-np.sum(values[values > 0] * np.log(values[values > 0])))

def key_stability(posterior):
    posterior = np.asarray(posterior, float)
    if posterior.ndim != 1 or len(posterior) < 2:
        raise ValueError('key posterior needs at least two key hypotheses')
    h = entropy(posterior)
    return component(1 - h, float(np.clip(1 - h / math.log(len(posterior)), 0, 1)), '12/16', entropy_nats=h, hypotheses=len(posterior), mapping='1-H/log(K)')

def note_diversity(pitches, durations, protocol):
    pitches, durations = (np.asarray(pitches), np.asarray(durations, float))
    if not len(pitches) or pitches.shape != durations.shape:
        raise ValueError('diversity needs aligned nonempty note arrays')
    if not np.isfinite(durations).all() or (durations <= 0).any() or (durations > protocol.max_note_duration_seconds).any():
        raise ValueError('duration outside declared diversity support')
    if (pitches < 0).any() or (pitches > 127).any() or (not np.equal(pitches, pitches.astype(int)).all()):
        raise ValueError('invalid pitch')
    pitch_entropy = entropy(np.unique(pitches, return_counts=True)[1])
    bins = np.ceil(durations / protocol.duration_quantum_seconds).astype(int)
    duration_entropy = entropy(np.unique(bins, return_counts=True)[1])
    upper = math.log(128) + math.log(math.ceil(protocol.max_note_duration_seconds / protocol.duration_quantum_seconds))
    raw = pitch_entropy + duration_entropy
    return component(raw, raw / upper, '17', pitch_entropy_nats=pitch_entropy, duration_entropy_nats=duration_entropy, fixed_support_log_bound=upper)

def consistency(onset_symbolic, onset_audio, energy_symbolic, energy_audio, protocol):

    def score():
        terms = []
        for coefficient, left, right in ((protocol.onset_coefficient, onset_symbolic, onset_audio), (protocol.envelope_coefficient, energy_symbolic, energy_audio)):
            if coefficient:
                terms.append(coefficient * correlation(left, right))
        return sum(terms)
    return corr_component(score, '21/22', onset_coefficient=protocol.onset_coefficient, envelope_coefficient=protocol.envelope_coefficient, scope='onset and energy only; no pitch/instrument/track fidelity assertion')

def symbolic_components(*, pitches, durations, key_posterior, violations, visual_curve, symbolic_curve, visual_boundaries, phrase_boundaries, protocol):
    if not 0 <= violations <= len(pitches) or not len(pitches):
        raise ValueError('playability requires notes and unique violating-note count')
    phrase = onset_f1(visual_boundaries, phrase_boundaries, protocol.boundary_tolerance_seconds)
    return {'beat': corr_component(lambda: lagged_ncc(visual_curve, symbolic_curve, sample_hz=protocol.analysis_hz, max_lag_seconds=protocol.max_lag_seconds)[0], '14'), 'phrase': component(phrase, phrase, '15', tolerance_seconds=protocol.boundary_tolerance_seconds, matching='one-to-one', empty_empty=0.0), 'key': key_stability(key_posterior), 'diversity': note_diversity(pitches, durations, protocol), 'playability': component(1 - violations / len(pitches), 1 - violations / len(pitches), '18', violations=violations, notes=len(pitches))}

def aggregate(components, weights, *, undefined='error'):
    if undefined not in {'error', 'zero'} or not weights:
        raise ValueError('explicit undefined=error|zero and nonempty weights required')
    values = np.array(list(weights.values()), float)
    if not np.isfinite(values).all() or (values < 0).any() or (not np.isclose(values.sum(), 1)):
        raise ValueError('objective weights must be a simplex')
    missing, reward = ([], 0.0)
    for name, weight in weights.items():
        if not weight:
            continue
        item = components[name]
        if item['status'] != 'ok':
            missing.append(name)
        else:
            reward += weight * item['reward']
    if missing and undefined == 'error':
        raise ValueError(f'unavailable reward measurements: {missing}')
    return {'reward': float(reward), 'weights': weights, 'undefined_policy': undefined, 'zeroed_unavailable': missing}

def template_key_posterior(pitch_class_mass, temperature=0.1):
    chroma = np.asarray(pitch_class_mass, float)
    if chroma.shape != (12,) or not np.isfinite(chroma).all() or (chroma < 0).any() or (chroma.sum() <= 0):
        raise ValueError('key inference needs nonzero pitch-class mass')
    scales = ((0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10))
    scores = [sum((chroma[(tonic + degree) % 12] for degree in scale)) / chroma.sum() for scale in scales for tonic in range(12)]
    logits = np.asarray(scores) / temperature
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum()

def event_evidence(events, duration, protocol):
    events = [e.to_dict() if hasattr(e, 'to_dict') else dict(e) for e in events]
    if not events or not math.isfinite(duration) or duration <= 0:
        raise ValueError('nonempty score and positive duration required')
    grid = np.arange(math.ceil(duration * protocol.analysis_hz)) / protocol.analysis_hz
    onset, energy, chroma = (np.zeros(len(grid)), np.zeros(len(grid)), np.zeros(12))
    violations, spans = (set(), [])
    for i, event in enumerate(events):
        pitch, start, length = (event['pitch'], float(event['onset']), float(event['duration']))
        velocity = float(event.get('velocity', 0.7))
        if not np.isfinite([pitch, start, length, velocity]).all() or not 0 <= pitch <= 127 or int(pitch) != pitch or (start < 0) or (length <= 0) or (not 0 <= velocity <= 1):
            raise ValueError('invalid realized note event')
        if start >= duration:
            continue
        end = min(duration, start + length)
        onset += velocity * np.exp(-0.5 * ((grid - start) / protocol.onset_width_seconds) ** 2)
        energy[(grid >= start) & (grid < end)] += velocity ** 2
        chroma[int(pitch) % 12] += (end - start) * max(velocity, 1e-06)
        spans.append((start, end))
        if not protocol.min_pitch <= pitch <= protocol.max_pitch:
            violations.add(i)
        overlap = [j for j, other in enumerate(events) if other.get('track_id', '') == event.get('track_id', '') and other['onset'] <= start < other['onset'] + other['duration']]
        if len(overlap) > protocol.max_polyphony_per_track:
            violations.update(overlap)
    if not spans:
        raise ValueError('no score notes within the declared duration')
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    boundaries = [left[1] for left, right in zip(merged, merged[1:]) if right[0] - left[1] >= protocol.phrase_silence_seconds]
    return {'grid': grid, 'onset': onset, 'energy': np.sqrt(energy), 'chroma': chroma, 'pitches': [e['pitch'] for e in events], 'durations': [e['duration'] for e in events], 'violations': len(violations), 'phrase_boundaries': boundaries, 'phrase_extractor': 'global silence-gap boundaries; reconstruction convention, not annotated phrase truth'}

def visual_evidence(pack, grid):
    evidence = pack.get('frame_features', pack)
    times = np.asarray(evidence['times'], float)
    motion = np.asarray(evidence['motion_intensity'], float)
    cuts = np.asarray(evidence['cut_flags'], float)
    novelty = np.asarray(evidence['visual_novelty'], float)
    if min(len(times), len(grid)) < 3 or any((x.shape != times.shape for x in (motion, cuts, novelty))) or (not np.isfinite([times, motion, cuts, novelty]).all()) or (np.diff(times) <= 0).any():
        raise ValueError('measured visual times and aligned motion/cut/novelty arrays required')
    if ((cuts < 0) | (cuts > 1)).any() or times[0] > grid[0] + 1e-06:
        raise ValueError('invalid visual event stream or missing start coverage')
    return {'accent': np.interp(grid, times, 0.65 * motion + 0.35 * cuts), 'novelty': np.interp(grid, times, novelty), 'boundaries': times[cuts > 0.5].tolist(), 'source_times': times.tolist(), 'interpolation': 'linear; last sample held to clip end; no new timing precision'}

def score_symbolic(events, visual_pack, duration, protocol=None):
    protocol = protocol or RewardProtocol()
    score = event_evidence(events, duration, protocol)
    visual = visual_evidence(visual_pack, score['grid'])
    components = symbolic_components(pitches=score['pitches'], durations=score['durations'], key_posterior=template_key_posterior(score['chroma'], protocol.key_temperature), violations=score['violations'], visual_curve=visual['accent'], symbolic_curve=score['onset'], visual_boundaries=visual['boundaries'], phrase_boundaries=score['phrase_boundaries'], protocol=protocol)
    return {'version': VERSION, 'components': components, 'protocol': asdict(protocol), 'evidence': {'visual': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in visual.items()}, 'phrase_boundaries': score['phrase_boundaries'], 'phrase_extractor': score['phrase_extractor'], 'key_model': '24 scale-membership templates'}}

def score_audio(events, visual_pack, audio_path, duration, protocol=None, *, emotion_measurement=None):
    from reward.evidence import extract_audio_features
    from experiments.provenance import sha256_file
    import librosa
    protocol = protocol or RewardProtocol()
    score = event_evidence(events, duration, protocol)
    visual = visual_evidence(visual_pack, score['grid'])
    audio = extract_audio_features(str(audio_path))
    if abs(audio['duration'] - duration) > 0.05:
        raise ValueError('reward audio duration differs from condition')
    aud_on = np.interp(score['grid'], audio['frame_times'], audio['onset'])
    aud_en = np.interp(score['grid'], audio['frame_times'], audio['rms_absolute'])
    import soundfile as sf
    wave, sr = sf.read(audio_path, always_2d=True)
    wave = wave.mean(1)
    chroma = librosa.feature.chroma_stft(y=wave, sr=sr, n_fft=2048, hop_length=max(1, round(sr / protocol.summary_hz)))
    novelty = np.linalg.norm(np.diff(chroma, axis=1, prepend=chroma[:, :1]), axis=0)
    novelty_times = np.arange(len(novelty)) * max(1, round(sr / protocol.summary_hz)) / sr
    audible = [i / protocol.analysis_hz for i in range(1, len(aud_en)) if aud_en[i - 1] >= 0.001 > aud_en[i]]
    cut = onset_f1(visual['boundaries'], audible, protocol.boundary_tolerance_seconds)
    try:
        tonal = key_stability(template_key_posterior(chroma.sum(1), protocol.key_temperature))
    except ValueError as exc:
        tonal = {'status': 'unavailable', 'raw': None, 'reward': None, 'equation': '12', 'reason': str(exc)}
    components = {'motion': corr_component(lambda: 0.5 * (correlation(score['onset'], visual['accent']) + correlation(aud_on, visual['accent'])), '19/20'), 'cut': component(cut, cut, '11', audio_boundary_rule='absolute RMS falling below 1e-3'), 'tonal': tonal, 'novelty': corr_component(lambda: correlation(np.interp(novelty_times, score['grid'], visual['novelty']), novelty), '11', measurement='2Hz chroma differences vs visual novelty'), 'consistency': consistency(score['onset'], aud_on, score['energy'], aud_en, protocol), 'emotion': {'status': 'unavailable', 'reward': None, 'raw': None, 'equation': '11', 'reason': 'requires an independent configured VA evaluator for this generated waveform'}}
    if emotion_measurement is not None:
        if not emotion_measurement.get('checkpoint') or emotion_measurement.get('audio_sha256') != sha256_file(audio_path):
            raise ValueError('emotion measurement must identify evaluator and current generated audio SHA-256')
        if emotion_measurement.get('status') == 'unavailable':
            components['emotion'].update(reason=emotion_measurement['reason'], evidence=emotion_measurement)
        else:
            raw = float(emotion_measurement['correlation'])
            if not -1 <= raw <= 1:
                raise ValueError('invalid emotion correlation')
            components['emotion'] = component(raw, (raw + 1) / 2, '11', **emotion_measurement)
    return {'version': VERSION, 'components': components, 'protocol': asdict(protocol), 'evidence': {'audio_sha256': sha256_file(audio_path), 'visual': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in visual.items()}, 'audio_frontend': '20ms Hann/10ms spectral-flux and absolute RMS; not madmom', 'tonal_model': '24 scale-membership templates on waveform chroma', 'audio_window_curves': {'times': score['grid'].tolist(), 'symbolic_onset': score['onset'].tolist(), 'audio_onset': aud_on.tolist(), 'symbolic_energy': score['energy'].tolist(), 'audio_energy': aud_en.tolist()}}}
