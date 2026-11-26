from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .blueprint_schema import (
    BlueprintGlobal,
    BlueprintSegment,
    BlueprintTrack,
    MusicBlueprint,
    VALID_REGISTERS,
    VALID_SECTIONS,
    VALID_TRACK_ROLES,
)


TRACK_ROLE_ALIASES = {
    "rhythm": "drums",
    "percussion": "drums",
}


def _clip01(value: Any, default: float = 0.5) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        value_f = default
    return max(0.0, min(1.0, value_f))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _infer_energy(prior_5d: Dict[str, float]) -> float:
    motion = _clip01(prior_5d.get("motion_intensity", 0.5))
    cuts = _clip01(prior_5d.get("cut_density", 0.5))
    novelty = _clip01(prior_5d.get("visual_novelty", 0.5))
    return _clip01(0.5 * motion + 0.3 * cuts + 0.2 * novelty)


def _infer_emotion(prior_5d: Dict[str, float]) -> str:
    brightness = _clip01(prior_5d.get("brightness", 0.5))
    warmth = _clip01(prior_5d.get("color_warmth", 0.5))
    novelty = _clip01(prior_5d.get("visual_novelty", 0.5))
    motion = _clip01(prior_5d.get("motion_intensity", 0.5))
    if motion > 0.7 and novelty > 0.6:
        return "tense"
    if brightness > 0.65 and warmth > 0.55:
        return "warm"
    if brightness < 0.4 and novelty > 0.45:
        return "dark"
    if motion < 0.35 and warmth > 0.5:
        return "calm"
    return "neutral"


def _infer_section(index: int, total: int) -> str:
    if total <= 1:
        return "intro"
    ratio = index / max(1, total - 1)
    if ratio < 0.2:
        return "intro"
    if ratio < 0.4:
        return "build"
    if ratio < 0.7:
        return "sustain"
    if ratio < 0.9:
        return "climax"
    return "outro"


def _infer_tempo(prior_5d: Dict[str, float], energy: float) -> int:
    motion = _clip01(prior_5d.get("motion_intensity", 0.5))
    cuts = _clip01(prior_5d.get("cut_density", 0.5))
    bpm = 84 + 42 * energy + 18 * motion + 12 * cuts
    return max(70, min(160, int(round(bpm))))


def _infer_key(prior_5d: Dict[str, float], emotion: str) -> str:
    brightness = _clip01(prior_5d.get("brightness", 0.5))
    warmth = _clip01(prior_5d.get("color_warmth", 0.5))
    if emotion in {"tense", "dark"}:
        return "Gmin"
    if brightness > 0.65 and warmth > 0.55:
        return "Cmaj"
    if warmth > 0.6:
        return "Fmaj"
    return "Dmin"


def _infer_chords(key: str, emotion: str, energy: float) -> List[str]:
    root = key[:-3] if key.endswith("maj") else key[:-3] if key.endswith("min") else key
    major = key.endswith("maj")
    if emotion == "tense":
        return [f"{root}m" if not major else root, "Eb", "F", "D"]
    if emotion == "warm":
        return [root, "Am", "F", "G"] if major else [f"{root}m", "Bb", "F", "C"]
    if energy > 0.7:
        return [root, "F", "C", "G"] if major else [f"{root}m", "Eb", "Bb", "D"]
    return [root, "F", "G", "C"] if major else [f"{root}m", "Eb", "F", "Gm"]


def _default_tracks(energy: float) -> List[BlueprintTrack]:
    density_mid = _clip01(energy)
    density_low = _clip01(0.75 * energy + 0.15)
    return [
        BlueprintTrack("melody", "mid", density_mid, "lead"),
        BlueprintTrack("bass", "low", density_low, "support"),
        BlueprintTrack("pad", "full", _clip01(0.65 * energy + 0.1), "texture"),
    ]


@dataclass
class MusicBlueprintParser:
    default_style_tags: Optional[List[str]] = None

    def parse(self, raw: Dict[str, Any], *, source: str = "llm") -> MusicBlueprint:
        global_raw = raw.get("global") or {}
        segments_raw = raw.get("segments") or []
        if not segments_raw:
            raise ValueError("blueprint must contain at least one segment")

        global_plan = BlueprintGlobal(
            tempo_bpm=max(60, min(180, _safe_int(global_raw.get("tempo_bpm"), 120))),
            meter=_safe_str(global_raw.get("meter"), "4/4"),
            global_key=_safe_str(global_raw.get("global_key"), "Cmaj"),
            style_tags=self._normalize_style_tags(global_raw.get("style_tags")),
        )

        segment_total = len(segments_raw)
        segments = [self._parse_segment(seg, idx, segment_total, global_plan) for idx, seg in enumerate(segments_raw)]
        blueprint = MusicBlueprint(global_plan=global_plan, segments=segments, source=source)
        from .timeline import validate_blueprint
        validate_blueprint(blueprint)
        return blueprint

    def _normalize_style_tags(self, value: Any) -> List[str]:
        if isinstance(value, list):
            tags = [_safe_str(item, "") for item in value]
            return [item for item in tags if item]
        if value is None:
            return list(self.default_style_tags or ["cinematic"])
        one = _safe_str(value, "")
        return [one] if one else list(self.default_style_tags or ["cinematic"])

    def _parse_segment(
        self,
        raw: Dict[str, Any],
        index: int,
        segment_total: int,
        global_plan: BlueprintGlobal,
    ) -> BlueprintSegment:
        section = _safe_str(raw.get("section"), _infer_section(index, segment_total)).lower()
        if section not in VALID_SECTIONS:
            section = _infer_section(index, segment_total)

        tracks = self._parse_tracks(raw.get("tracks"), raw.get("energy"))
        prior = raw.get("video_prior_5d") or {}
        normalized_prior = {
            "motion_intensity": _clip01(prior.get("motion_intensity", 0.5)),
            "cut_density": _clip01(prior.get("cut_density", 0.5)),
            "brightness": _clip01(prior.get("brightness", 0.5)),
            "color_warmth": _clip01(prior.get("color_warmth", 0.5)),
            "visual_novelty": _clip01(prior.get("visual_novelty", 0.5)),
        }
        return BlueprintSegment(
            segment_id=_safe_int(raw.get("segment_id"), index),
            start_time=float(raw.get("start_time", index * 4.0)),
            end_time=float(raw.get("end_time", (index + 1) * 4.0)),
            energy=_clip01(raw.get("energy", 0.5)),
            emotion=_safe_str(raw.get("emotion"), "neutral"),
            section=section,
            chords=self._parse_chords(raw.get("chords"), global_plan.global_key),
            tracks=tracks,
            tempo_bpm=max(60, min(180, _safe_int(raw.get("tempo_bpm"), global_plan.tempo_bpm))),
            meter=_safe_str(raw.get("meter"), global_plan.meter),
            global_key=_safe_str(raw.get("global_key"), global_plan.global_key),
            video_prior_5d=normalized_prior,
        )

    def _parse_tracks(self, raw_tracks: Any, energy: Any) -> List[BlueprintTrack]:
        if not isinstance(raw_tracks, list) or not raw_tracks:
            return _default_tracks(_clip01(energy))
        tracks: List[BlueprintTrack] = []
        for raw in raw_tracks:
            if not isinstance(raw, dict):
                continue
            role = _safe_str(raw.get("role"), "melody").lower()
            role = TRACK_ROLE_ALIASES.get(role, role)
            if role not in VALID_TRACK_ROLES:
                role = "melody"
            register = _safe_str(raw.get("register"), "mid").lower()
            if register not in VALID_REGISTERS:
                register = "mid"
            tracks.append(
                BlueprintTrack(
                    role=role,
                    register=register,
                    density=_clip01(raw.get("density", energy if energy is not None else 0.5)),
                    phrase_function=_safe_str(raw.get("phrase_function"), "support"),
                    section_role=_safe_str(raw.get("section_role"), "support"),
                    track_id=str(raw.get("track_id", "")),
                    midi_program=_safe_int(raw.get("midi_program"), 0),
                    pitch_range=list(raw.get("pitch_range") or []),
                )
            )
        return tracks or _default_tracks(_clip01(energy))

    def _parse_chords(self, raw: Any, global_key: str) -> List[str]:
        if isinstance(raw, list):
            chords = [_safe_str(item, "") for item in raw]
            chords = [item for item in chords if item]
            return chords
        if raw:
            return [_safe_str(raw, global_key)]
        if global_key.endswith("maj"):
            return [global_key[:-3]]
        if global_key.endswith("min"):
            return [global_key[:-3]+"m"]
        return [global_key]


def blueprint_from_segment_priors(
    prior_data: Dict[str, Any],
    *,
    style_tags: Optional[List[str]] = None,
    meter: str = "4/4",
) -> MusicBlueprint:
    segments_raw = []
    priors = prior_data.get("prior_5d") or []
    inferred_keys: List[str] = []
    global_bpms: List[int] = []
    for index, prior in enumerate(priors):
        normalized_prior = {
            "motion_intensity": _clip01(prior.get("motion_intensity", 0.5)),
            "cut_density": _clip01(prior.get("cut_density", 0.5)),
            "brightness": _clip01(prior.get("brightness", 0.5)),
            "color_warmth": _clip01(prior.get("color_warmth", 0.5)),
            "visual_novelty": _clip01(prior.get("visual_novelty", 0.5)),
        }
        energy = _infer_energy(normalized_prior)
        emotion = _infer_emotion(normalized_prior)
        key = _infer_key(normalized_prior, emotion)
        inferred_keys.append(key)
        tempo_bpm = _infer_tempo(normalized_prior, energy)
        global_bpms.append(tempo_bpm)
        segments_raw.append(
            {
                "segment_id": index,
                "start_time": float(prior.get("start_time", index * 4.0)),
                "end_time": float(prior.get("end_time", (index + 1) * 4.0)),
                "energy": energy,
                "emotion": emotion,
                "section": _infer_section(index, len(priors)),
                "chords": _infer_chords(key, emotion, energy),
                "tracks": [track.to_dict() for track in _default_tracks(energy)],
                "tempo_bpm": tempo_bpm,
                "meter": meter,
                "global_key": key,
                "video_prior_5d": normalized_prior,
            }
        )
    if not segments_raw:
        raise ValueError("prior data does not contain any segments")
    parser = MusicBlueprintParser(default_style_tags=style_tags or ["cinematic", "video-conditioned"])
    inferred_key = Counter(inferred_keys).most_common(1)[0][0] if inferred_keys else "Cmaj"
    avg_bpm = int(round(sum(global_bpms) / max(1, len(global_bpms))))
    raw = {
        "global": {
            "tempo_bpm": avg_bpm,
            "meter": meter,
            "global_key": inferred_key,
            "style_tags": style_tags or ["cinematic", "video-conditioned"],
        },
        "segments": segments_raw,
    }
    return parser.parse(raw, source="heuristic-prior")


def save_blueprint_json(blueprint: MusicBlueprint, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blueprint.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_blueprint_json(path: str) -> MusicBlueprint:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    parser = MusicBlueprintParser()
    return parser.parse(raw, source=_safe_str(raw.get("source"), "json"))
