"""Prepare local CoMA datasets with identity-aware splits and explicit annotations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from experiments.provenance import sha256_file
from symbolic.events import NoteEvent, export_midi
from planning.timeline import TempoMap

SPLITS = ("train", "validation", "test")
IDENTITIES = ("movie_id", "music_work_id", "video_sha256", "target_audio_sha256", "source_midi_sha256")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in rows), encoding="utf-8")


def new_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_index(path):
    path = Path(path).resolve()
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("each index row requires a unique sample_id")
        seen.add(sid)
        for key in ("video", "audio", "blueprint", "symbolic_outputs", "abc", "reward_inputs"):
            if row.get(key):
                value = Path(row[key])
                value = (value if value.is_absolute() else path.parent/value).resolve()
                if not value.is_file():
                    raise FileNotFoundError(value)
                row[key] = str(value)
        rows.append(row)
    if not rows:
        raise ValueError("empty dataset index")
    return rows


def grouped_split(rows, *, seed=2026, ratios=(.8, .1, .1)):
    """Split connected identity groups; preserve explicit official partitions."""
    if len(ratios) != 3 or not np.isfinite(ratios).all() or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("three positive split ratios summing to one required")
    rows = [dict(r) for r in sorted(rows, key=lambda r: r["sample_id"])]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample_id")
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        for kind in IDENTITIES:
            if row.get(kind):
                key = (kind, row[kind])
                if key in owners:
                    parents[find(i)] = find(owners[key])
                owners[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    counts = Counter()
    pending = []
    for ids in groups.values():
        specified = {rows[i]["split"] for i in ids if rows[i].get("split")}
        if len(specified) > 1 or specified-set(SPLITS):
            raise ValueError("connected movie/work/file group crosses declared partitions")
        if specified:
            split = specified.pop()
            for i in ids:
                rows[i]["split"] = split
            counts[split] += len(ids)
        else:
            pending.append(ids)
    random.Random(seed).shuffle(pending)
    pending.sort(key=len, reverse=True)
    targets = dict(zip(SPLITS, [r*len(rows) for r in ratios]))
    for ids in pending:
        # Minimize the change in squared count error, without breaking a group.
        split = min(SPLITS, key=lambda s: (counts[s]+len(ids)-targets[s])**2-(counts[s]-targets[s])**2)
        for i in ids:
            rows[i]["split"] = split
        counts[split] += len(ids)
    return rows, {"seed": seed, "group_count": len(groups), "counts": dict(counts),
                  "target_counts": targets, "identity_fields": list(IDENTITIES),
                  "exact_counts_guaranteed": False, "group_integrity_precedes_count_matching": True}


def midi_events(path):
    """Decode type 0/1 MIDI using tempo events, channels and sustain pedal."""
    import mido
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("asynchronous/SMPTE MIDI is unsupported")
    changes = {0: 500000}
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                changes[tick] = message.tempo
    points, elapsed, previous, tempo = [], 0., 0, 500000
    for tick, new_tempo in sorted(changes.items()):
        elapsed += mido.tick2second(tick-previous, midi.ticks_per_beat, tempo)
        points.append((elapsed, mido.tempo2bpm(new_tempo)))
        previous, tempo = tick, new_tempo
    clock = TempoMap(points)
    events = []
    for track_index, track in enumerate(midi.tracks):
        tick, programs = 0, defaultdict(int)
        active, sustained, pedal = defaultdict(deque), defaultdict(list), defaultdict(bool)
        def finish(note, end):
            start, pitch, velocity, program, channel = note
            if end > start:
                role = "drums" if channel == 9 else "bass" if 32 <= program <= 39 else "melody"
                events.append(NoteEvent(f"track{track_index:02d}_ch{channel:02d}", pitch, start, end-start,
                                        velocity/127, program, role))
        for message in track:
            tick += message.time
            now = clock.beats_to_seconds(tick/midi.ticks_per_beat)
            channel = getattr(message, "channel", 0)
            if message.type == "program_change":
                programs[channel] = message.program
            elif message.type == "note_on" and message.velocity > 0:
                active[channel, message.note].append((now, message.note, message.velocity, programs[channel], channel))
            elif message.type in {"note_off", "note_on"}:
                notes = active[channel, message.note]
                if notes:
                    note = notes.popleft()
                    if pedal[channel]:
                        sustained[channel].append(note)
                    else:
                        finish(note, now)
            elif message.type == "control_change" and message.control == 64:
                pedal[channel] = message.value >= 64
                if not pedal[channel]:
                    for note in sustained.pop(channel, []):
                        finish(note, now)
        end = clock.beats_to_seconds(tick/midi.ticks_per_beat)
        for notes in sustained.values():
            for note in notes:
                finish(note, end)
        if any(active.values()):
            raise ValueError(f"unclosed MIDI note in track {track_index}")
    if not events:
        raise ValueError("MIDI has no sounding notes")
    return sorted(events, key=lambda e: (e.onset, e.track_id, e.pitch)), clock


def clip_events(events, start, duration):
    return [replace(e, onset=max(start, e.onset)-start,
                    duration=min(start+duration, e.onset+e.duration)-max(start, e.onset))
            for e in events if e.onset < start+duration and e.onset+e.duration > start]


def activity_seconds(events):
    end, total = 0., 0.
    for e in sorted(events, key=lambda e: e.onset):
        right = e.onset+e.duration
        total += max(0., right-max(end, e.onset))
        end = max(end, right)
    return total


def _pitch_abc(midi_pitch):
    names = ("=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B")
    octave = midi_pitch//12-5
    return names[midi_pitch % 12] + ("'"*octave if octave >= 0 else ","*(-octave))


def abc_voices(events, *, bpm=120, duration=15.):
    """Greedy voice separation with synchronous chords; 1/96 quarter-note grid.

    Each voice is a separate MuPT example compatible with per-track inference.
    Audio conditioning retains all original, unquantized events.
    """
    units = 96*bpm/60
    groups = defaultdict(list)
    for event in events:
        start = round(event.onset*units)
        end = max(start+1, round((event.onset+event.duration)*units))
        groups[start, end].append(event.pitch)
    lanes = []
    for (start, end), pitches in sorted(groups.items()):
        lane = next((line for line in lanes if line[-1][1] <= start), None)
        if lane is None:
            lane = []
            lanes.append(lane)
        lane.append((start, end, sorted(pitches)))
    def length(ticks):
        value = Fraction(ticks, 48)
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    for lane in lanes:
        tokens, cursor = [], 0
        for start, end, pitches in lane:
            if start > cursor:
                tokens.append("z"+length(start-cursor))
            notes = "".join(_pitch_abc(p) for p in pitches)
            tokens.append((f"[{notes}]" if len(pitches) > 1 else notes)+length(end-start))
            cursor = end
        tail = round(duration*units)-cursor
        if tail > 0:
            tokens.append("z"+length(tail))
        yield " ".join(tokens)+" |\n"


def _crop_audio(path, destination, start, duration):
    from scipy.signal import resample_poly
    with sf.SoundFile(path) as source:
        sr = source.samplerate
        source.seek(round(start*sr))
        wave = source.read(round(duration*sr), always_2d=True).mean(1)
    if len(wave) != round(duration*sr) or not np.isfinite(wave).all():
        raise ValueError("incomplete/nonfinite source audio window")
    divisor = math.gcd(sr, 32000)
    wave = resample_poly(wave, 32000//divisor, sr//divisor) if sr != 32000 else wave
    sf.write(destination, wave[:round(duration*32000)], 32000, subtype="FLOAT")


def prepare_slakh(source, output, *, duration=15., max_windows=32, min_tracks=2, min_onsets=8, min_activity=10.,
                  split_limits=(20000, 2000, 1000), seed=2026):
    if not 0 < duration <= 30 or max_windows < 1 or min_tracks < 1 or min_onsets < 1 or not 0 <= min_activity <= duration:
        raise ValueError("invalid Slakh window/filter configuration")
    if len(split_limits) != 3 or any(n < 1 for n in split_limits):
        raise ValueError("three positive split window caps required")
    root = new_directory(output)
    source = Path(source).resolve()
    audio_rows, symbolic_rows, rejected, source_rows = [], [], [], []
    for partition in sorted(source.iterdir()):
        split = {"valid": "validation", "val": "validation"}.get(partition.name, partition.name)
        if split not in SPLITS or not partition.is_dir():
            continue
        for track in sorted(partition.iterdir()):
            if not track.is_dir():
                continue
            midi = track/"all_src.mid"
            audio = next((track/name for name in ("mix.flac", "mix.wav") if (track/name).is_file()), None)
            if not midi.is_file() or audio is None:
                rejected.append({"source": str(track), "reason": "missing all_src.mid or mix.flac/mix.wav"})
                continue
            try:
                events, clock = midi_events(midi)
                info = sf.info(audio)
                source_row = {"sample_id": f"slakh:{track.name}", "split": split,
                              "source_midi_sha256": sha256_file(midi), "target_audio_sha256": sha256_file(audio),
                              "music_work_id": f"slakh:{track.name}"}
                source_rows.append(source_row)
                selected = 0
                for window in range(math.floor(info.duration/duration)):
                    if selected >= max_windows:
                        break
                    start = window*duration
                    notes = clip_events(events, start, duration)
                    tracks = sorted({e.track_id for e in notes})
                    reason = ("too_few_tracks" if len(tracks) < min_tracks else
                              "too_few_onsets" if len({round(e.onset, 6) for e in notes}) < min_onsets else
                              "insufficient_activity" if activity_seconds(notes) < min_activity else None)
                    if reason:
                        rejected.append({"source": str(track), "window_start": start, "reason": reason})
                        continue
                    selected += 1
                    destination = root/split/track.name/f"window_{window:04d}"
                    destination.mkdir(parents=True)
                    _crop_audio(audio, destination/"audio.wav", start, duration)
                    tempo = [(0., next(bpm for t, bpm in reversed(clock.points) if t <= start))]
                    tempo.extend((t-start, bpm) for t, bpm in clock.points if start < t < start+duration)
                    export_midi(notes, destination/"score.mid", TempoMap(tempo))
                    # SFT controls are score-derived; they are never visual PPO evidence.
                    specs = []
                    for tid in tracks:
                        subset = [e for e in notes if e.track_id == tid]
                        specs.append({"track_id": tid, "role": subset[0].role, "register": "full",
                                      "density": min(1., len({e.onset for e in subset})/(16*duration)),
                                      "phrase_function": "source_part", "section_role": "support",
                                      "midi_program": subset[0].program,
                                      "pitch_range": [min(e.pitch for e in subset), max(e.pitch for e in subset)]})
                    bpm = max(60, min(180, round(tempo[0][1])))
                    segment = {"segment_id": 0, "start_time": 0., "end_time": duration, "energy": float(np.mean([e.velocity for e in notes])),
                               "emotion": "unspecified", "section": "sustain", "chords": [], "tracks": specs,
                               "tempo_bpm": bpm, "meter": "4/4", "global_key": "C", "video_prior_5d": {}}
                    blueprint = {"global": {"tempo_bpm": bpm, "meter": "4/4", "global_key": "C", "style_tags": ["instrumental"]},
                                 "segments": [segment], "source": "slakh_score_derived_sft_controls_not_video_annotations"}
                    write_json(destination/"blueprint.json", blueprint)
                    outputs = [{"segment_id": 0, "track_id": tid, "events": [e.to_dict() for e in notes if e.track_id == tid]} for tid in tracks]
                    write_json(destination/"symbolic_outputs.json", outputs)
                    row = {**source_row, "sample_id": f"slakh:{track.name}:{window:04d}", "dataset": "Slakh2100",
                           "audio": str(destination/"audio.wav"), "blueprint": str(destination/"blueprint.json"),
                           "symbolic_outputs": str(destination/"symbolic_outputs.json"), "duration_seconds": duration,
                           "source_start_seconds": start, "target_audio_sha256": sha256_file(destination/"audio.wav"),
                           "source_audio": str(audio), "source_midi": str(midi), "condition_source": "score_derived_sft_only"}
                    audio_rows.append(row)
                    for spec in specs:
                        subset = [e for e in notes if e.track_id == spec["track_id"]]
                        for voice, body in enumerate(abc_voices(subset, bpm=bpm, duration=duration)):
                            name = f"{spec['track_id']}_voice{voice:02d}"
                            prefix = (f"X:1\nT:{name}\nM:4/4\nL:1/8\nQ:1/4={bpm}\n"
                                      f"% role={spec['role']}; register=full; density={spec['density']:.3f}; "
                                      "phrase=source_part; section=sustain; chords=\nK:C\n")
                            (destination/f"{name}.abc").write_text(prefix+body)
                            symbolic_rows.append({**row, "sample_id": row["sample_id"]+":"+name,
                                                  "clip_sample_id": row["sample_id"],
                                                  "abc": str(destination/f"{name}.abc"), "prefix": prefix})
            except (ValueError, OSError, EOFError) as exc:
                rejected.append({"source": str(track), "reason": str(exc)})
    # Official partitions must also agree for duplicate source files.
    grouped_split(source_rows)
    rng = random.Random(seed)
    eligible_counts = dict(Counter(r["split"] for r in audio_rows))
    selected_rows = []
    for split, limit in zip(SPLITS, split_limits):
        available = [r for r in audio_rows if r["split"] == split]
        selected_rows.extend(rng.sample(available, limit) if len(available) > limit else available)
    audio_rows = sorted(selected_rows, key=lambda r: r["sample_id"])
    selected_ids = {r["sample_id"] for r in audio_rows}
    symbolic_rows = [r for r in symbolic_rows if r["clip_sample_id"] in selected_ids]
    for split in SPLITS:
        write_jsonl(root/f"audio_{split}.jsonl", [r for r in audio_rows if r["split"] == split])
        write_jsonl(root/f"symbolic_{split}.jsonl", [r for r in symbolic_rows if r["split"] == split])
    report = {"kind": "slakh", "status": "prepared" if audio_rows else "empty", "source": str(source),
              "audio_examples": len(audio_rows), "symbolic_examples": len(symbolic_rows), "rejected": rejected,
              "window_seconds": duration, "max_windows_per_track": max_windows,
              "split_window_caps": dict(zip(SPLITS, split_limits)), "eligible_window_counts": eligible_counts,
              "selected_window_counts": dict(Counter(r["split"] for r in audio_rows)), "selection_seed": seed,
              "filters": {"min_tracks": min_tracks, "min_onsets": min_onsets, "min_activity_seconds": min_activity},
              "symbolic_preprocessing": "score-derived SFT controls; synchronous chord grouping and greedy voice separation; 1/96-quarter grid; C is a chromatic notation reference, not inferred key",
              "audio_conditioning": "all original MIDI events in seconds, including sustain and tempo changes",
              "limitation": "no video/reward labels; not a PPO manifest; perceptual fingerprint deduplication still required"}
    write_json(root/"dataset_report.json", report)
    if not audio_rows:
        raise ValueError("no Slakh windows passed filters; see dataset_report.json")
    return report


def mix_audio(slakh_manifest, musicpro_manifest, output, *, seed=2026):
    """Create an equal-example SFT mixture with explicit oversampling identities."""
    from training.data import read_training_manifest
    sources = [read_training_manifest(path, "audio-sft") for path in (slakh_manifest, musicpro_manifest)]
    root = new_directory(output)
    count = max(map(len, sources))
    rng, rows = random.Random(seed), []
    for source_name, source in zip(("slakh", "musicpro"), sources):
        pool = []
        while len(pool) < count:
            cycle = list(source)
            rng.shuffle(cycle)
            pool.extend(cycle)
        for i, row in enumerate(pool[:count]):
            rows.append({**row, "sample_id": f"{source_name}:draw{i:06d}:{row['sample_id']}",
                         "source_sample_id": row["sample_id"], "mixture_source": source_name})
    rng.shuffle(rows)
    write_jsonl(root/"audio_train.jsonl", rows)
    report = {"kind": "equal_audio_sft_mixture", "seed": seed, "examples_per_source": count,
              "source_counts": [len(s) for s in sources], "oversampling": "shuffled cycles with explicit source_sample_id",
              "scope": "SFT only; source MusicPro symbolic_outputs must come from the declared frozen transcription model"}
    write_json(root/"dataset_report.json", report)
    return report


def prepare_index(index, output, *, kind, seed=2026, extract_visual=False, exclude_manifest=None):
    from experiments.data_protocol import audit_dataset
    rows = read_index(index)
    hashes = {}
    for row in rows:
        for key in ("movie_id", "dataset") + (("music_work_id",) if kind == "musicpro" else ()):
            if not row.get(key):
                raise ValueError(f"{row['sample_id']}: explicit {key} required")
        if not row.get("video"):
            raise ValueError("video path required")
        for key, digest in (("video", "video_sha256"), ("audio", "target_audio_sha256")):
            if row.get(key):
                hashes.setdefault(row[key], None)
                if hashes[row[key]] is None:
                    hashes[row[key]] = sha256_file(row[key])
                if row.get(digest) and row[digest] != hashes[row[key]]:
                    raise ValueError(f"{row['sample_id']}: {digest} mismatch")
                row[digest] = hashes[row[key]]
        if kind == "musicpro":
            if not row.get("audio"):
                raise ValueError("MusicPro index requires the held-out target audio path")
            row.setdefault("input_condition", "V")
        else:
            if isinstance(row.get("prior_5d"), dict):
                from vedio.cineparse import LABELS
                if set(row["prior_5d"]) != set(LABELS):
                    raise ValueError("CineParse prior_5d keys must match the five descriptor names")
            labels = np.asarray(list(row["prior_5d"].values()) if isinstance(row.get("prior_5d"), dict) else row.get("prior_5d"), float)
            if labels.shape != (5,) or not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
                raise ValueError("CineParse needs five annotated labels in [0,1]")
            if not row.get("label_source") or not row.get("label_protocol"):
                raise ValueError("CineParse requires explicit label_source and label_protocol")
            start, end = float(row["start_time"]), float(row["end_time"])
            if not math.isfinite(start+end) or not 0 <= start < end:
                raise ValueError("invalid CineParse annotation interval")
            hits = row.get("hit_times")
            if hits is not None and (hits != sorted(set(hits)) or any(not start <= t < end for t in hits)):
                raise ValueError("hit timestamps must be unique, sorted and inside the annotation interval")
    rows, split_report = grouped_split(rows, seed=seed)
    excluded = []
    if exclude_manifest:
        held_out = [r for r in read_index(exclude_manifest) if r.get("split") in {"validation", "test"}]
        forbidden = {(key, r[key]) for r in held_out for key in IDENTITIES if r.get(key)}
        kept = []
        for row in rows:
            if row["split"] == "train" and any((key, row.get(key)) in forbidden for key in IDENTITIES):
                excluded.append(row["sample_id"])
            else:
                kept.append(row)
        rows = kept
    audit = audit_dataset(rows) if kind == "musicpro" else {"valid": True, "scope": "connected identity split; annotated label ranges/times"}
    if not audit["valid"]:
        raise ValueError(f"dataset audit failed: {audit['errors']}")
    root = new_directory(output)
    if extract_visual:
        from vedio.prior_feature_pack import extract_prior_5d
        for i, row in enumerate(rows):
            # Native-rate measured curves are separate from supervised CineParse targets.
            prior = extract_prior_5d(row["video"], sample_fps=1e6)
            destination = root/f"sample_{i:06d}"
            destination.mkdir()
            write_json(destination/"prior.json", prior)
            evidence = {"measurement_source": "native_frame_rate_opencv_measurements", "visual": prior,
                        "video_sha256": row["video_sha256"]}
            if row.get("emotion_target"):
                evidence["emotion_target"] = row["emotion_target"]
            write_json(destination/"reward_inputs.json", evidence)
            row.update(prior=str(destination/"prior.json"), reward_inputs=str(destination/"reward_inputs.json"))
    write_jsonl(root/"all.jsonl", rows)
    for split in SPLITS:
        write_jsonl(root/f"{split}.jsonl", [r for r in rows if r["split"] == split])
        if kind == "musicpro":
            write_jsonl(root/f"guidance_{split}.jsonl", [{k: v for k, v in r.items() if k not in {"audio", "abc", "symbolic_outputs"}}
                for r in rows if r["split"] == split and r.get("blueprint") and r.get("reward_inputs")])
            write_jsonl(root/f"audio_{split}.jsonl", [r for r in rows if r["split"] == split and r.get("blueprint") and r.get("symbolic_outputs")])
            write_jsonl(root/f"symbolic_ppo_{split}.jsonl", [{k: v for k, v in r.items() if k not in {"audio", "abc", "symbolic_outputs"}}
                for r in rows if r["split"] == split and r.get("prefix") and r.get("reward_inputs") and r.get("duration_seconds")])
    report = {"kind": kind, "source_index_sha256": sha256_file(index), "split": split_report,
              "counts_after_exclusion": dict(Counter(r["split"] for r in rows)), "excluded_from_visual_training": excluded,
              "audit": audit, "extract_visual": extract_visual,
              "limitation": "file hashes detect exact duplicates; canonical work IDs and acoustic fingerprints must be supplied/checked separately; no automatic Composer or transcription"}
    write_json(root/"dataset_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    slakh = commands.add_parser("slakh", help="official partition folders containing Track*/all_src.mid and mix.flac")
    slakh.add_argument("--source", required=True)
    slakh.add_argument("--output-dir", required=True)
    slakh.add_argument("--window-seconds", type=float, default=15.)
    slakh.add_argument("--max-windows", type=int, default=32)
    slakh.add_argument("--min-tracks", type=int, default=2)
    slakh.add_argument("--min-onsets", type=int, default=8)
    slakh.add_argument("--min-activity", type=float, default=10.)
    slakh.add_argument("--split-window-caps", nargs=3, type=int, default=[20000, 2000, 1000])
    slakh.add_argument("--seed", type=int, default=2026)
    mixture = commands.add_parser("mix-audio", help="equal Slakh/MusicPro SFT example counts")
    mixture.add_argument("--slakh-manifest", required=True)
    mixture.add_argument("--musicpro-manifest", required=True)
    mixture.add_argument("--output-dir", required=True)
    mixture.add_argument("--seed", type=int, default=2026)
    for kind in ("musicpro", "cineparse"):
        command = commands.add_parser(kind, help="prepare a JSONL index with explicit identities/annotations")
        command.add_argument("--index", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--seed", type=int, default=2026)
        command.add_argument("--extract-visual", action="store_true")
        command.add_argument("--exclude-manifest", help="exclude held-out movie/work identities from visual training")
    args = parser.parse_args(argv)
    if args.kind == "slakh":
        report = prepare_slakh(args.source, args.output_dir, duration=args.window_seconds, max_windows=args.max_windows,
                               min_tracks=args.min_tracks, min_onsets=args.min_onsets, min_activity=args.min_activity,
                               split_limits=args.split_window_caps, seed=args.seed)
    elif args.kind == "mix-audio":
        report = mix_audio(args.slakh_manifest, args.musicpro_manifest, args.output_dir, seed=args.seed)
    else:
        report = prepare_index(args.index, args.output_dir, kind=args.kind, seed=args.seed,
                               extract_visual=args.extract_visual, exclude_manifest=args.exclude_manifest)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rejected", "split"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
