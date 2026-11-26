"""Strict boundary validation for LLM output; never silently repairs bad fields."""
import math
import re
from .blueprint_schema import VALID_REGISTERS, VALID_SECTIONS, VALID_TRACK_ROLES


def validate_raw_blueprint(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("global"), dict):
        raise ValueError("blueprint.global must be an object")
    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("blueprint.segments must be a nonempty list")
    def number(obj, key, low, high):
        value = obj.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
    def musical(obj):
        number(obj, "tempo_bpm", 60, 180)
        if not re.fullmatch(r"[1-9]\d*/(?:1|2|4|8|16|32)", str(obj.get("meter", ""))):
            raise ValueError("invalid meter")
        if not re.fullmatch(r"[A-G](?:#|b)?(?:maj|min|m)?", str(obj.get("global_key", ""))):
            raise ValueError("invalid global_key")
    musical(raw["global"])
    ids = set()
    for s in segments:
        if not isinstance(s, dict):
            raise ValueError("each segment must be an object")
        number(s, "segment_id", 0, 1e9)
        if int(s["segment_id"]) != s["segment_id"] or s["segment_id"] in ids:
            raise ValueError("segment IDs must be unique integers")
        ids.add(s["segment_id"])
        number(s, "start_time", 0, 1e9); number(s, "end_time", 0, 1e9)
        if s["end_time"] <= s["start_time"]:
            raise ValueError("segment end must exceed start")
        number(s, "energy", 0, 1)
        musical({**raw["global"], **s})
        if s.get("section") not in VALID_SECTIONS:
            raise ValueError("invalid section")
        tracks = s.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            raise ValueError("tracks must be a nonempty list")
        for t in tracks:
            if t.get("role") not in VALID_TRACK_ROLES or t.get("register") not in VALID_REGISTERS:
                raise ValueError("invalid track role/register")
            number(t, "density", 0, 1)
            if not t.get("phrase_function") or not t.get("section_role"):
                raise ValueError("track phrase_function and section_role required")


def bind_visual_priors(raw, prior_data):
    """The LLM must preserve observation intervals; visual evidence is never invented."""
    priors = prior_data.get("prior_5d", [])
    if len(raw["segments"]) != len(priors):
        raise ValueError("Composer changed the number of observation windows")
    for s, p in zip(raw["segments"], priors):
        if any(abs(s[k]-p[k]) > 1e-5 for k in ("start_time", "end_time")):
            raise ValueError("Composer changed an observation interval")
        s["video_prior_5d"] = {k: p[k] for k in ("motion_intensity", "cut_density", "brightness", "color_warmth", "visual_novelty")}
    return raw
