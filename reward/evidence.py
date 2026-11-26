from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_control_to_prompt
from planning.blueprint_schema import MusicBlueprint
from planning.blueprint_to_abc import blueprint_to_mupt_prefixes


NOTE_RE = re.compile(r"(?<![A-Za-z])[\^_=]*([A-Ga-gzZxX])([,']*)(\d+)?(?:/(\d+))?")
KEY_TO_INDEX = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    length = min(len(x), len(y))
    if length < 2:
        return 0.0
    x = x[:length]
    y = y[:length]
    if float(np.ptp(x)) < 1e-6 or float(np.ptp(y)) < 1e-6:
        return 1.0 - min(1.0, abs(float(np.mean(x)) - float(np.mean(y))))
    return _clip01(0.5 * (float(np.corrcoef(x, y)[0, 1]) + 1.0))


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(values) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(values) == 1:
        return np.full(length, float(values[0]), dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, length)
    return np.interp(new_x, old_x, values).astype(np.float32)


def _align_feature(values: np.ndarray, length: int, *, reducer: str = "mean") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) <= length:
        return _resample(values, length)
    edges = np.linspace(0, len(values), length + 1, dtype=np.int32)
    aligned = np.zeros(length, dtype=np.float32)
    for index in range(length):
        chunk = values[edges[index] : max(edges[index] + 1, edges[index + 1])]
        aligned[index] = float(np.max(chunk) if reducer == "max" else np.mean(chunk))
    return aligned


def _visual_curves(prior: Dict[str, float], length: int) -> Dict[str, np.ndarray]:
    motion = _clip01(prior.get("motion_intensity", 0.5))
    cuts = _clip01(prior.get("cut_density", 0.5))
    novelty = _clip01(prior.get("visual_novelty", 0.5))
    brightness = _clip01(prior.get("brightness", 0.5))
    warmth = _clip01(prior.get("color_warmth", 0.5))
    # Segment descriptors have no within-segment event times. Do not manufacture
    # sinusoidal beats or a synthetic late cut from pooled evidence.
    accent = np.full(length, 0.65*motion + 0.35*cuts, dtype=np.float32)
    return {
        "accent": accent[:length],
        "brightness": np.full(length, brightness, dtype=np.float32),
        "warmth": np.full(length, warmth, dtype=np.float32),
        "novelty": np.full(length, novelty, dtype=np.float32),
    }


def _key_posterior(key: str, length: int) -> np.ndarray:
    normalized = str(key or "C").upper().replace("MAJ", "").replace("MIN", "").replace("M", "")
    index = KEY_TO_INDEX.get(normalized, 0)
    posterior = np.full((max(1, length), 12), 1e-3, dtype=np.float32)
    posterior[:, index] = 1.0
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior


def extract_event_features(events, *, duration, key, source="abc"):
    from symbolic.events import NoteEvent
    events = [e.to_dict() if isinstance(e, NoteEvent) else e for e in events]
    events = [e for e in events if 0 <= e["onset"] < duration]
    length = max(8, int(np.ceil(duration*100)))
    onset = np.zeros(length, dtype=np.float32); velocity = np.zeros(length, dtype=np.float32)
    pitches=[]; durations=[]
    for e in events:
        left=min(length-1,int(e["onset"]*100)); right=min(length,max(left+1,int((e["onset"]+e["duration"])*100)))
        onset[left]=1; velocity[left:right] += e.get("velocity",.7)
        pitches.append(e["pitch"]); durations.append(e["duration"])
    if velocity.max()>0: velocity/=velocity.max()
    # Estimate pitch-class distributions from actual notes, not a one-hot desired key.
    chroma=np.full(12,1e-3,dtype=np.float32)
    for p,d in zip(pitches,durations): chroma[p%12]+=d
    chroma/=chroma.sum()
    return {"onset":onset,"velocity":velocity,"phrase_boundaries":[],
            "key_posterior":np.tile(chroma,(length,1)),"pitches":np.asarray(pitches,dtype=np.int32),
            "durations":np.asarray(durations,dtype=np.float32),"violations":sum(not 21<=p<=108 for p in pitches),
            "total_notes":len(pitches),"note_density":len(pitches)/max(duration,.1),"duration":float(duration),"source":source}


def extract_symbolic_features(abc_text: str, *, duration: float, key: str) -> Dict[str, Any]:
    from symbolic.events import abc_to_events
    normalized=str(abc_text or "").replace("<n>","\n")
    tunes=[t for t in re.split(r"(?m)(?=^X:)",normalized) if t.strip()]
    events=[]; offset=0.
    for tune in tunes:
        parsed=abc_to_events(tune,start_time=offset)
        events.extend(parsed)
        if parsed:offset=max(e.onset+e.duration for e in parsed)
    return extract_event_features(events,duration=duration,key=key)


def extract_audio_features(audio_path: str) -> Dict[str, Any]:
    import soundfile as sf

    waveform, sample_rate = sf.read(audio_path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        raise ValueError(f"empty audio file: {audio_path}")
    frame_length = max(256, int(sample_rate * 0.02))
    hop_length = max(128, int(sample_rate * 0.01))
    frames = []
    for start in range(0, max(1, len(waveform) - frame_length + 1), hop_length):
        frame = waveform[start : start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frames.append(frame)
    frame_array = np.stack(frames)
    rms = np.sqrt(np.mean(frame_array ** 2, axis=1) + 1e-8)
    spectra = np.abs(np.fft.rfft(frame_array * np.hanning(frame_length), axis=1))
    flux = np.maximum(0.0, np.diff(spectra, axis=0, prepend=spectra[:1])).mean(axis=1)
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    centroid = (spectra * frequencies[None, :]).sum(axis=1) / (spectra.sum(axis=1) + 1e-8)
    hf_mask = frequencies >= 4000.0
    hf_ratio = spectra[:, hf_mask].sum(axis=1) / (spectra.sum(axis=1) + 1e-8)
    def norm(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        lo, hi = float(np.min(values)), float(np.max(values))
        return np.zeros_like(values) if hi - lo < 1e-8 else (values - lo) / (hi - lo)
    return {
        "onset": norm(flux),
        "rms": norm(rms),
        "rms_absolute": rms.astype(np.float32),
        "frame_times": np.arange(len(rms))*hop_length/float(sample_rate),
        "analysis_hz": sample_rate/float(hop_length),
        "spectral_centroid": norm(centroid),
        "hf_ratio": norm(hf_ratio),
        "sample_rate": int(sample_rate),
        "duration": len(waveform) / float(sample_rate),
        "source": "waveform",
    }


def build_reward_evidence(
    *,
    prior: Dict[str, float],
    control: Dict[str, Any],
    symbolic_features: Dict[str, Any],
    audio_features: Optional[Dict[str, Any]] = None,
    chord_count: int = 0,
    track_count: int = 0,
    temporal_alignment_freq: float = 2.0,
) -> Dict[str, Any]:
    if temporal_alignment_freq <= 0:
        raise ValueError("temporal_alignment_freq must be positive")
    raw_symbolic_onset = np.asarray(symbolic_features.get("onset", []), dtype=np.float32)
    raw_audio_onset = np.asarray((audio_features or {}).get("onset", raw_symbolic_onset), dtype=np.float32)
    control_duration = max(0.0, float(control.get("end_time", 0.0)) - float(control.get("start_time", 0.0)))
    duration = max(
        0.1,
        float(symbolic_features.get("duration", 0.0)),
        float((audio_features or {}).get("duration", 0.0)),
        control_duration,
    )
    length = max(2, int(math.ceil(duration * temporal_alignment_freq)))
    symbolic_onset = _align_feature(raw_symbolic_onset, length, reducer="max")
    audio_onset = _align_feature(raw_audio_onset, length, reducer="max")
    symbolic_velocity = _align_feature(np.asarray(symbolic_features.get("velocity", raw_symbolic_onset)), length)
    audio_rms = _align_feature(np.asarray((audio_features or {}).get("rms", symbolic_velocity)), length)
    centroid = _align_feature(np.asarray((audio_features or {}).get("spectral_centroid", [control.get("brightness_hint", prior.get("brightness", 0.5))])), length)
    hf_ratio = _align_feature(np.asarray((audio_features or {}).get("hf_ratio", centroid)), length)
    visual = _visual_curves(prior, length)
    source_symbolic_length = max(1, len(raw_symbolic_onset))
    phrase_boundaries = [
        int(round((float(boundary) / max(1, source_symbolic_length - 1)) * (length - 1)))
        for boundary in symbolic_features.get("phrase_boundaries", [])
    ]
    visual_boundaries = [length - 1] if prior.get("visual_novelty", 0.0) > 0.55 else []

    target_density = 1.5 + 8.0 * (0.55 * prior.get("motion_intensity", 0.5) + 0.45 * prior.get("cut_density", 0.5))
    note_density = float(symbolic_features.get("note_density", 0.0))
    density_match = 1.0 - min(1.0, abs(note_density - target_density) / max(1.0, target_density))
    tonal_stability = 1.0
    key_posterior = np.asarray(symbolic_features.get("key_posterior", _key_posterior(control.get("key", "C"), length)))
    if key_posterior.size:
        probabilities = np.clip(key_posterior, 1e-8, 1.0)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1).mean() / math.log(probabilities.shape[1])
        tonal_stability = _clip01(1.0 - entropy)
    harmonic_progression = _clip01(0.35 + 0.15 * min(4, chord_count) + 0.05 * min(4, track_count))
    structural_symmetry = _clip01(0.5 * (1.0 if phrase_boundaries else 0.5) + 0.5 * min(1.0, track_count / 4.0))
    motion_rhythm = _safe_corr(visual["accent"], 0.5 * symbolic_onset + 0.5 * audio_onset)
    brightness_score = 1.0 - min(1.0, abs(float(np.mean(centroid)) - prior.get("brightness", 0.5)))
    warmth_score = 1.0 - min(1.0, abs((1.0 - float(np.mean(hf_ratio))) - prior.get("color_warmth", 0.5)))
    color_tone = _clip01(0.5 * (brightness_score + warmth_score))
    novelty_thematic = _clip01(0.5 * density_match + 0.5 * (1.0 if phrase_boundaries or prior.get("visual_novelty", 0.0) < 0.5 else 0.4))

    return {
        "symbolic_data": {
            "beat": {"g_vis": visual["accent"], "g_sym": symbolic_onset},
            "phrase": {"B_vis": visual_boundaries, "B_sym": phrase_boundaries},
            "key": {"key_posterior": key_posterior},
            "diversity": {
                "pitches": np.asarray(symbolic_features.get("pitches", [])),
                "durations": np.asarray(symbolic_features.get("durations", [])),
            },
            "playability": {
                "violations": int(symbolic_features.get("violations", 0)),
                "total_notes": int(symbolic_features.get("total_notes", 0)),
            },
        },
        "consistency_data": {} if audio_features is None else {
            "onset": {"e_sym": symbolic_onset, "e_aud": audio_onset},
            "dynamic": {"rms_aud": audio_rms, "vel_sym": symbolic_velocity},
            "color": {
                "C_spec": centroid,
                "w": visual["warmth"],
                "hf_ratio": hf_ratio,
                "b": visual["brightness"],
            },
        },
        "visual_data": {
            "motion_rhythm": motion_rhythm,
            "color_tone": color_tone,
            "novelty_thematic": novelty_thematic,
        },
        "musicality_data": {
            "tonal_stability": tonal_stability,
            "harmonic_progression": harmonic_progression,
            "rhythmic_density": density_match,
            "structural_symmetry": structural_symmetry,
        },
        "evidence_source": {
            "symbolic": symbolic_features.get("source", "control"),
            "audio": (audio_features or {}).get("source", "control"),
        },
        "temporal_alignment_hz": float(temporal_alignment_freq),
        "visual_evidence_scope": "pooled_segment_proxy_no_event_timing",
        "candidate_conditioning_verified": False,
    }


class RewardEvidenceEvaluator:
    def __init__(
        self,
        *,
        symbolic_outputs: Optional[List[Dict[str, Any]]] = None,
        audio_manifest: Optional[Dict[str, Any]] = None,
        temporal_alignment_freq: float = 2.0,
    ) -> None:
        if temporal_alignment_freq <= 0:
            raise ValueError("temporal_alignment_freq must be positive")
        self.temporal_alignment_freq = float(temporal_alignment_freq)
        self.events_by_segment: Dict[int, List[Dict[str, Any]]] = {}
        self.symbolic_by_segment: Dict[int, List[str]] = {}
        for item in symbolic_outputs or []:
            self.events_by_segment.setdefault(int(item.get("segment_id",0)),[]).extend(item.get("events",[]))
            self.symbolic_by_segment.setdefault(int(item.get("segment_id", 0)), []).append(str(item.get("abc", "")))
        self.audio_paths = list((audio_manifest or {}).get("segment_paths", []))

    def evaluate(self, blueprint: MusicBlueprint, segment_index: int, control: Dict[str, Any]) -> Dict[str, Any]:
        segment = blueprint.segments[segment_index]
        abc_items = self.symbolic_by_segment.get(segment.segment_id)
        if abc_items:
            abc_text = "\n".join(abc_items)
            symbolic_source = "abc"
        else:
            one_segment = copy.deepcopy(blueprint)
            one_segment.segments = [copy.deepcopy(segment)]
            abc_text = "\n".join(blueprint_to_mupt_prefixes(one_segment))
            symbolic_source = "blueprint_prefix"
        if self.events_by_segment.get(segment.segment_id):
            local_events=[dict(e,onset=e["onset"]-segment.start_time) for e in self.events_by_segment[segment.segment_id]]
            symbolic=extract_event_features(local_events,duration=segment.end_time-segment.start_time,key=segment.global_key,source="realized_events")
        else:
            symbolic = extract_symbolic_features(abc_text,duration=max(0.1,segment.end_time-segment.start_time),key=segment.global_key)
        symbolic["source"] = symbolic_source
        audio = None
        if segment_index < len(self.audio_paths) and Path(self.audio_paths[segment_index]).exists():
            audio = extract_audio_features(self.audio_paths[segment_index])
        return build_reward_evidence(
            prior=segment.video_prior_5d,
            control=control,
            symbolic_features=symbolic,
            audio_features=audio,
            chord_count=len(segment.chords),
            track_count=len(segment.tracks),
            temporal_alignment_freq=self.temporal_alignment_freq,
        )


class ModelBackedRewardEvaluator(RewardEvidenceEvaluator):
    def __init__(
        self,
        *,
        output_dir: str,
        mupt_model_dir: str,
        musicgen_model_dir: str,
        mupt_device: Optional[str] = None,
        musicgen_device: Optional[str] = None,
        seed: int = 123,
        temporal_alignment_freq: float = 2.0,
    ) -> None:
        super().__init__(temporal_alignment_freq=temporal_alignment_freq)
        self.output_dir = Path(output_dir)
        self.mupt_model_dir = mupt_model_dir
        self.musicgen_model_dir = musicgen_model_dir
        self.mupt_device = mupt_device
        self.musicgen_device = musicgen_device
        self.seed = seed
        self.cache: Dict[str, Dict[str, Any]] = {}

    def evaluate(self, blueprint: MusicBlueprint, segment_index: int, control: Dict[str, Any]) -> Dict[str, Any]:
        import hashlib
        import json

        cache_key = hashlib.sha1(
            json.dumps({"segment": blueprint.segments[segment_index].to_dict(), "control": control}, sort_keys=True).encode()
        ).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        from audio.musicgen_renderer import MusicGenRendererSession
        from track.melody.infer import generate

        segment = copy.deepcopy(blueprint.segments[segment_index])
        one_segment = copy.deepcopy(blueprint)
        one_segment.segments = [segment]
        prefix = blueprint_to_mupt_prefixes(one_segment)[0]
        abc_text = generate(
            prefix=prefix,
            max_new_tokens=384,
            temperature=0.8,
            top_p=0.95,
            seed=self.seed + segment_index,
            model_dir=self.mupt_model_dir,
            device=self.mupt_device,
        )
        symbolic = extract_symbolic_features(
            abc_text,
            duration=max(0.1, segment.end_time - segment.start_time),
            key=segment.global_key,
        )
        enriched_control = dict(control)
        enriched_control["symbolic_summary"] = {"prompt_hint": f"ABC motif with {symbolic['note_density']:.2f} notes per second"}
        prompt = render_control_to_prompt(enriched_control)
        renderer = MusicGenRendererSession.get(
            model_dir=self.musicgen_model_dir,
            device=self.musicgen_device,
        )
        audio_path = renderer.render_one(
            prompt,
            output_path=str(self.output_dir / cache_key / "candidate.wav"),
            seed=self.seed + segment_index,
            duration=max(0.1,segment.end_time-segment.start_time),
        )
        audio = extract_audio_features(audio_path)
        evidence = build_reward_evidence(
            prior=segment.video_prior_5d,
            control=control,
            symbolic_features=symbolic,
            audio_features=audio,
            chord_count=len(segment.chords),
            track_count=len(segment.tracks),
            temporal_alignment_freq=self.temporal_alignment_freq,
        )
        evidence["evidence_source"] = {"symbolic": "mupt", "audio": "musicgen_text_only"}
        evidence["candidate_conditioning_verified"] = True
        self.cache[cache_key] = evidence
        return evidence


__all__ = [
    "ModelBackedRewardEvaluator",
    "RewardEvidenceEvaluator",
    "build_reward_evidence",
    "extract_audio_features",
    "extract_symbolic_features",
]
