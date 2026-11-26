from __future__ import annotations

import re
from typing import Dict, List

from planning.blueprint_schema import MusicBlueprint


def _energy_label(energy: float) -> str:
    if energy < 0.4:
        return "low"
    if energy < 0.75:
        return "medium"
    return "high"


def _texture_label(track_roles: List[str], energy: float) -> str:
    has_drums = "drums" in track_roles
    has_pad = "pad" in track_roles
    has_bass = "bass" in track_roles
    if has_drums and has_bass and energy > 0.7:
        return "dense rhythmic orchestration"
    if has_pad and energy < 0.45:
        return "sustained ambient texture"
    if has_bass:
        return "grounded layered accompaniment"
    return "focused melodic texture"


def _summarize_symbolic_abc(abc_items: List[str]) -> Dict[str, object]:
    abc_text = "\n".join(item for item in abc_items if item)
    if not abc_text.strip():
        return {}
    body_lines = [
        line for line in abc_text.replace("<n>", "\n").splitlines()
        if not re.match(r"^[A-Z]:", line.strip())
    ]
    body = " ".join(body_lines)
    notes = re.findall(r"(?<![A-Za-z])[\^_=]?[A-Ga-g][,']*", body)
    rests = re.findall(r"(?<![A-Za-z])[zZxX]", body)
    bars = max(1, body.count("|") // 2)
    high_notes = sum(1 for note in notes if note[-1:].islower() or "'" in note)
    low_notes = sum(1 for note in notes if "," in note)
    note_density = len(notes) / bars
    if note_density < 4:
        density_label = "sparse"
    elif note_density < 9:
        density_label = "moderate"
    else:
        density_label = "dense"
    if high_notes > low_notes * 1.5:
        register_label = "upper-register"
    elif low_notes > high_notes * 1.5:
        register_label = "lower-register"
    else:
        register_label = "mid-register"
    rest_ratio = len(rests) / max(1, len(notes) + len(rests))
    return {
        "abc": abc_text,
        "note_count": len(notes),
        "bar_count": bars,
        "note_density": float(note_density),
        "density_label": density_label,
        "register_label": register_label,
        "rest_ratio": float(rest_ratio),
        "prompt_hint": f"{density_label} {register_label} symbolic motif with {rest_ratio:.0%} rests",
    }


def _symbolic_by_segment(symbolic_outputs: List[Dict[str, object]] | None) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = {}
    for item in symbolic_outputs or []:
        segment_id = int(item.get("segment_id", 0))
        abc_text = str(item.get("abc", "")).strip()
        if abc_text:
            grouped.setdefault(segment_id, []).append(abc_text)
    return grouped


def blueprint_to_render_controls(
    blueprint: MusicBlueprint,
    symbolic_outputs: List[Dict[str, object]] | None = None,
    window_seconds: float = 0.4,
) -> List[Dict[str, object]]:
    controls: List[Dict[str, object]] = []
    grouped_symbolic = _symbolic_by_segment(symbolic_outputs)
    for segment in blueprint.segments:
        roles = [track.role for track in segment.tracks]
        symbolic_summary = _summarize_symbolic_abc(grouped_symbolic.get(segment.segment_id, []))
        duration = max(0.0, segment.end_time - segment.start_time)
        window_count = max(1, int(round(duration / max(0.2, min(0.5, window_seconds)))))
        window_duration = duration / window_count if duration > 0 else max(0.2, min(0.5, window_seconds))
        render_windows = [
            {
                "window_index": window_index,
                "start_time": segment.start_time + window_index * window_duration,
                "end_time": segment.start_time + (window_index + 1) * window_duration,
                "tempo_bpm": segment.tempo_bpm,
                "energy": segment.energy,
                "brightness_hint": segment.video_prior_5d.get("brightness", 0.5),
                "texture_bias": 0.0,
            }
            for window_index in range(window_count)
        ]
        control = {
            "segment_id": segment.segment_id,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "tempo_bpm": segment.tempo_bpm,
            "meter": segment.meter,
            "key": segment.global_key,
            "energy": segment.energy,
            "energy_label": _energy_label(segment.energy),
            "emotion": segment.emotion,
            "section": segment.section,
            "style_tags": list(blueprint.global_plan.style_tags),
            "chords": list(segment.chords),
            "track_roles": roles,
            "tracks": [track.to_dict() for track in segment.tracks],
            "texture_hint": _texture_label(roles, segment.energy),
            "video_prior_5d": dict(segment.video_prior_5d),
            "symbolic_summary": symbolic_summary,
            "render_window_seconds": window_duration,
            "render_windows": render_windows,
        }
        controls.append(control)
    return controls
