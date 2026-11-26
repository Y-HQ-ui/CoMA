from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import re

from planning.timeline import TempoMap


@dataclass(frozen=True)
class NoteEvent:
    track_id: str
    pitch: int
    onset: float
    duration: float
    velocity: float = 0.7
    program: int = 0
    role: str = "melody"

    def __post_init__(self):
        if not 0 <= self.pitch <= 127 or not 0 <= self.program <= 127:
            raise ValueError("pitch and program must be valid MIDI values")
        if not all(math.isfinite(x) for x in (self.onset, self.duration, self.velocity)) or self.onset < 0 or self.duration <= 0 or not 0 <= self.velocity <= 1:
            raise ValueError("invalid note timing/velocity")

    def to_dict(self):
        return asdict(self)


def abc_to_events(abc: str, *, tempo_map: TempoMap | None = None, start_time=0.0,
                  track_id="", role="melody", program=0) -> list[NoteEvent]:
    """Parse actual ABC durations, keys, accidentals, chords, voices and ties.

    An explicit blueprint tempo map takes precedence over ABC tempo marks. With
    no override, all ABC tempo changes are integrated. No time stretching is used.
    Repeats are kept as written (not unfolded); exported protocol records this.
    """
    from music21 import abcFormat, stream
    from .notation import decode_mupt
    text, notation = decode_mupt(abc)
    if not re.search(r"(?m)^K:", text):
        raise ValueError("ABC requires a K: header")
    handler = abcFormat.ABCFile().readstr(text)
    # music21 otherwise coerces an unrecognized note token to C with a warning.
    # Treat that as a failed model sample, not fabricated musical evidence.
    for token in handler.tokens:
        notes = token.subTokens if isinstance(token,abcFormat.ABCChord) else [token]
        for note in notes:
            if isinstance(note,abcFormat.ABCNote):
                _, pitched = note._splitChordSymbols(note.src)
                try: note.getPitchName(pitched)
                except abcFormat.ABCHandlerException as exc:
                    raise ValueError(f"invalid ABC note token {note.src!r}; refusing pitch substitution") from exc
    score = (abcFormat.translate.abcToStreamOpus(handler) if handler.definesReferenceNumbers()
             else abcFormat.translate.abcToStreamScore(handler))
    if isinstance(score, stream.Opus):
        if len(score.scores) != 1:
            raise ValueError("multiple ABC tunes require separate time origins")
        score = score.scores[0]
    if notation["synchronized"] and len(score.parts) > 1:
        offsets = [tuple(round(float(m.offset), 8) for m in p.getElementsByClass("Measure")) for p in score.parts]
        if len(set(offsets)) != 1:
            raise ValueError("MuPT voice bars have inconsistent durations; cannot silently align them")
    score = score.stripTies(inPlace=False)
    boundaries = score.metronomeMarkBoundaries()
    def abc_seconds(beat):
        seconds = 0.0
        for left, right, mark in boundaries:
            if beat <= left:
                break
            seconds += max(0, min(beat, right)-left)*60/float(mark.getQuarterBPM() or 120)
        return seconds
    origin = tempo_map.seconds_to_beats(start_time) if tempo_map else 0
    def to_seconds(beat):
        return tempo_map.beats_to_seconds(origin+beat) if tempo_map else start_time+abc_seconds(beat)
    parts = list(score.parts) or [score]
    events = []
    for i, part in enumerate(parts):
        tid = track_id if len(parts) == 1 and track_id else f"{track_id or 'voice'}:{i+1}"
        for note in part.flatten().notes:
            duration_beats = float(note.quarterLength)
            if duration_beats <= 0:  # grace notes need an explicit performance convention
                continue
            onset = to_seconds(float(note.offset))
            end = to_seconds(float(note.offset)+duration_beats)
            velocity = float(note.volume.realized or 0.7)
            for pitch in note.pitches:
                events.append(NoteEvent(tid, int(pitch.midi), onset, end-onset, velocity, program, role))
    if not events and not list(score.recurse().notesAndRests):
        raise ValueError("ABC has no musical events")
    return sorted(events, key=lambda e: (e.onset, e.track_id, e.pitch))


REGISTERS = {"low": (36, 59), "mid": (48, 83), "high": (72, 96), "full": (21, 108)}


def constrain_events(events, track, *, start, end, total_duration):
    """Disclosed deterministic postprocessor: octave-fold range and cap density.

    Density in [0,1] is a cap of 0..16 onset groups/second, not a guarantee of
    that many generated notes. Notes starting in a span may sustain across it.
    """
    low, high = track.pitch_range or REGISTERS[track.register]
    selected = [e for e in events if start <= e.onset < end and e.onset < total_duration]
    onsets = sorted({e.onset for e in selected})
    limit = int(math.ceil(16*track.density*(end-start))) if track.density > 0 else 0
    if len(onsets) > limit:
        import numpy as np
        keep = {onsets[i] for i in np.linspace(0, len(onsets)-1, limit, dtype=int)} if limit else set()
        selected = [e for e in selected if e.onset in keep]
    result = []
    for e in selected:
        pitch = e.pitch
        candidates = [p for p in range(low, high+1) if p % 12 == pitch % 12]
        pitch = min(candidates, key=lambda p: abs(p-e.pitch)) if candidates else min(high, max(low, pitch))
        result.append(replace(e, pitch=pitch, duration=min(e.duration, total_duration-e.onset),
                              track_id=track.track_id or track.role, role=track.role, program=track.midi_program))
    return result


def export_midi(events, path, tempo_map=None):
    """Write a multitrack SMF with a true tempo track and absolute note times."""
    import mido
    from pathlib import Path
    mapping = tempo_map or TempoMap([(0, 120)])
    midi = mido.MidiFile(ticks_per_beat=480)
    conductor = mido.MidiTrack(); midi.tracks.append(conductor)
    last = 0
    for seconds, bpm in mapping.points:
        tick = round(mapping.seconds_to_beats(seconds)*480)
        conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=tick-last)); last = tick
    identities = sorted({e.track_id for e in events})
    melodic_channels = [c for c in range(16) if c != 9]
    if sum(any(e.role != "drums" for e in events if e.track_id == tid) for tid in identities) > 15:
        raise ValueError("MIDI supports at most 15 melodic channels")
    channel_index = 0
    for tid in identities:
        notes = [e for e in events if e.track_id == tid]
        channel = 9 if notes[0].role == "drums" else melodic_channels[channel_index]
        if channel != 9: channel_index += 1
        track = mido.MidiTrack(); midi.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=tid))
        track.append(mido.Message("program_change", program=notes[0].program, channel=channel))
        messages = []
        for e in notes:
            messages.extend([(round(mapping.seconds_to_beats(e.onset)*480), 1, e),
                             (round(mapping.seconds_to_beats(e.onset+e.duration)*480), 0, e)])
        last = 0
        for tick, on, e in sorted(messages, key=lambda x: (x[0], x[1], x[2].pitch)):
            track.append(mido.Message("note_on" if on else "note_off", note=e.pitch,
                                      velocity=max(1, round(e.velocity*127)) if on else 0,
                                      channel=channel, time=max(0, tick-last))); last = tick
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))
