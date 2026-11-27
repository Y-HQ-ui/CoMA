from __future__ import annotations
import copy
import math
from typing import Iterable
from .blueprint_schema import MusicBlueprint, VALID_REGISTERS, VALID_TRACK_ROLES

def validate_blueprint(blueprint: MusicBlueprint) -> None:
    if not blueprint.segments:
        raise ValueError('blueprint must contain segments')
    ids = set()
    for s in blueprint.segments:
        if s.segment_id in ids:
            raise ValueError(f'duplicate segment_id: {s.segment_id}')
        ids.add(s.segment_id)
        numbers = [s.start_time, s.end_time, s.energy, s.tempo_bpm, *s.video_prior_5d.values()]
        if not all((math.isfinite(float(x)) for x in numbers)):
            raise ValueError(f'segment {s.segment_id}: non-finite control')
        if s.start_time < 0 or s.end_time <= s.start_time:
            raise ValueError(f'segment {s.segment_id}: require 0 <= start < end')
        if not 20 <= s.tempo_bpm <= 300 or not 0 <= s.energy <= 1:
            raise ValueError(f'segment {s.segment_id}: invalid tempo/energy')
        seen = set()
        for t in s.tracks:
            if t.role not in VALID_TRACK_ROLES or t.register not in VALID_REGISTERS:
                raise ValueError('unknown track role/register')
            if not isinstance(t.midi_program, int) or not 0 <= t.midi_program <= 127:
                raise ValueError('track MIDI program must be an integer in [0,127]')
            identity = t.track_id or t.role
            if identity in seen:
                raise ValueError(f'segment {s.segment_id}: duplicate track identity {identity}')
            seen.add(identity)
            if not math.isfinite(t.density) or not 0 <= t.density <= 1:
                raise ValueError('track density must be finite and in [0,1]')
            if t.pitch_range and (len(t.pitch_range) != 2 or not 0 <= t.pitch_range[0] <= t.pitch_range[1] <= 127):
                raise ValueError('pitch_range must be [low,high] MIDI pitches')

def merge_overlapping_segments(blueprint: MusicBlueprint) -> MusicBlueprint:
    validate_blueprint(blueprint)
    source = sorted(blueprint.segments, key=lambda s: (s.start_time, s.segment_id))
    if source[0].start_time != 0:
        raise ValueError('timeline must start at zero; represent initial silence explicitly')
    if all((b.start_time >= a.end_time for a, b in zip(source, source[1:]))):
        if any((b.start_time > a.end_time + 1e-08 for a, b in zip(source, source[1:]))):
            raise ValueError('timeline has an uncovered gap')
        result = copy.deepcopy(blueprint)
        result.segments = copy.deepcopy(source)
        return result
    edges = sorted({v for s in source for v in (s.start_time, s.end_time)})
    result = copy.deepcopy(blueprint)
    result.segments = []
    for left, right in zip(edges, edges[1:]):
        mid = (left + right) / 2
        active = [s for s in source if s.start_time <= mid < s.end_time]
        if not active:
            raise ValueError(f'timeline gap at {left}..{right}')
        selected = min(active, key=lambda s: (abs((s.start_time + s.end_time) / 2 - mid), s.segment_id))
        out = copy.deepcopy(selected)
        out.segment_id = len(result.segments)
        out.start_time, out.end_time = (left, right)
        out.energy = sum((s.energy for s in active)) / len(active)
        out.tempo_bpm = round(sum((s.tempo_bpm for s in active)) / len(active))
        out.video_prior_5d = {k: sum((s.video_prior_5d.get(k, 0.5) for s in active)) / len(active) for k in selected.video_prior_5d}
        for track in out.tracks:
            matches = [t for s in active for t in s.tracks if (t.track_id or t.role) == (track.track_id or track.role)]
            track.density = sum((t.density for t in matches)) / len(matches)
        result.segments.append(out)
    result.source = blueprint.source + ':overlap-pooled'
    return result

class TempoMap:

    def __init__(self, points: Iterable[tuple[float, float]]):
        self.points = sorted(((float(t), float(bpm)) for t, bpm in points))
        if not self.points or self.points[0][0] != 0:
            raise ValueError('tempo map must begin at second zero')
        if any((not math.isfinite(t + bpm) or bpm <= 0 for t, bpm in self.points)):
            raise ValueError('invalid tempo point')
        if any((b[0] <= a[0] for a, b in zip(self.points, self.points[1:]))):
            raise ValueError('tempo times must be strictly increasing')
        self.beats = [0.0]
        for (t0, bpm), (t1, _) in zip(self.points, self.points[1:]):
            self.beats.append(self.beats[-1] + (t1 - t0) * bpm / 60)

    @classmethod
    def from_blueprint(cls, blueprint: MusicBlueprint):
        merged = merge_overlapping_segments(blueprint)
        return cls(((s.start_time, s.tempo_bpm) for s in merged.segments))

    def seconds_to_beats(self, seconds: float) -> float:
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError('seconds must be finite and nonnegative')
        i = max((i for i, (t, _) in enumerate(self.points) if t <= seconds))
        t, bpm = self.points[i]
        return self.beats[i] + (seconds - t) * bpm / 60

    def beats_to_seconds(self, beats: float) -> float:
        if beats < 0 or not math.isfinite(beats):
            raise ValueError('beats must be finite and nonnegative')
        i = max((i for i, b in enumerate(self.beats) if b <= beats))
        return self.points[i][0] + (beats - self.beats[i]) * 60 / self.points[i][1]
