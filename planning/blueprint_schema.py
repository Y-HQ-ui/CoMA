from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
VALID_SECTIONS = {'intro', 'build', 'verse', 'sustain', 'transition', 'chorus', 'climax', 'release', 'outro'}
VALID_TRACK_ROLES = {'melody', 'harmony', 'bass', 'drums', 'pad', 'arpeggio', 'fx'}
VALID_REGISTERS = {'low', 'mid', 'high', 'full'}

@dataclass
class BlueprintTrack:
    role: str
    register: str
    density: float
    phrase_function: str
    section_role: str = 'support'
    track_id: str = ''
    midi_program: int = 0
    pitch_range: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BlueprintSegment:
    segment_id: int
    start_time: float
    end_time: float
    energy: float
    emotion: str
    section: str
    chords: List[str] = field(default_factory=list)
    tracks: List[BlueprintTrack] = field(default_factory=list)
    tempo_bpm: int = 120
    meter: str = '4/4'
    global_key: str = 'Cmaj'
    video_prior_5d: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['tracks'] = [track.to_dict() for track in self.tracks]
        return data

@dataclass
class BlueprintGlobal:
    tempo_bpm: int
    meter: str
    global_key: str
    style_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MusicBlueprint:
    global_plan: BlueprintGlobal
    segments: List[BlueprintSegment]
    source: str = 'heuristic'

    def to_dict(self) -> Dict[str, Any]:
        return {'global': self.global_plan.to_dict(), 'segments': [segment.to_dict() for segment in self.segments], 'source': self.source}
