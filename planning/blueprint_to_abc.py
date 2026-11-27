from __future__ import annotations
from typing import Dict, List
from .blueprint_schema import MusicBlueprint

def _note_length_from_energy(energy: float) -> str:
    if energy < 0.45:
        return '1/4'
    if energy < 0.8:
        return '1/8'
    return '1/16'

def _seed_pattern(chord: str, energy: float, emotion: str) -> str:
    root = chord[0].upper() if chord else 'C'
    minor = 'm' in chord.lower()
    third = 'B' if root == 'G' and minor else 'E' if not minor else 'D'
    if emotion == 'tense':
        return f"{root}{third}{root}'{third}"
    if energy > 0.75:
        return f"{root}{third}{root}{third}{root}'{third}"
    if energy < 0.4:
        return f'{root}2 z2 {third}2 z2'
    return f'{root}{third}{root}{third}'

def blueprint_to_bar_contexts(blueprint: MusicBlueprint) -> List[Dict[str, object]]:
    contexts: List[Dict[str, object]] = []
    for segment in blueprint.segments:
        contexts.append({'bar_index_start': segment.segment_id + 1, 'bar_count': max(1, len(segment.chords)), 'bpm': segment.tempo_bpm, 'meter': segment.meter, 'section': segment.section, 'chords': list(segment.chords), 'energies': [segment.energy for _ in segment.chords] or [segment.energy], 'global_key': segment.global_key, 'video_prior_5d': dict(segment.video_prior_5d), 'tracks': [track.to_dict() for track in segment.tracks], 'emotion': segment.emotion})
    return contexts

def blueprint_to_mupt_prefixes(blueprint: MusicBlueprint) -> List[str]:
    prefixes: List[str] = []
    bar_index = 1
    for segment in blueprint.segments:
        base_length = _note_length_from_energy(segment.energy)
        chords = segment.chords or [segment.global_key]
        for chord in chords:
            seed = _seed_pattern(chord, segment.energy, segment.emotion)
            prefix = f'X:{bar_index}<n>L:{base_length}<n>Q:{segment.tempo_bpm}<n>M:{segment.meter}<n>K:{segment.global_key}<n>|:"{chord}" {seed}'
            prefixes.append(prefix)
            bar_index += 1
    return prefixes
