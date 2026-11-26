import json
from pathlib import Path
import tempfile
import unittest

import mido
import numpy as np
import soundfile as sf

from scripts.verify_dataset import make_slakh_fixture
from symbolic.events import NoteEvent, abc_to_events
from training.data import read_training_manifest, symbolic_example
from training.prepare_dataset import (grouped_split, midi_events, clip_events, activity_seconds,
                                      abc_voices, prepare_slakh, prepare_index, write_jsonl)


class DatasetPreparationTests(unittest.TestCase):
    def test_transitive_groups_and_determinism(self):
        rows = [{"sample_id": "a", "movie_id": "m1", "music_work_id": "w1"},
                {"sample_id": "b", "movie_id": "m2", "music_work_id": "w1"},
                {"sample_id": "c", "movie_id": "m2", "music_work_id": "w2"}]
        rows += [{"sample_id": f"extra{i}", "movie_id": f"extra{i}"} for i in range(20)]
        assigned, _ = grouped_split(rows)
        again, _ = grouped_split(list(reversed(rows)))
        self.assertEqual(assigned, again)
        self.assertEqual(len({r["split"] for r in assigned if r["sample_id"] in {"a", "b", "c"}}), 1)
        rows[0]["split"], rows[2]["split"] = "train", "test"
        with self.assertRaisesRegex(ValueError, "crosses"):
            grouped_split(rows)

    def test_exact_file_duplicates_stay_together(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            grouped_split([{"sample_id": "a", "video_sha256": "same", "split": "train"},
                           {"sample_id": "b", "video_sha256": "same", "split": "test"}])

    def test_tempo_and_pedal_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.extend([mido.MetaMessage("set_tempo", tempo=500000),
                          mido.Message("note_on", note=60, velocity=80),
                          mido.Message("control_change", control=64, value=127, time=240),
                          mido.MetaMessage("set_tempo", tempo=1000000, time=240),
                          mido.Message("note_off", note=60, time=240),
                          mido.Message("control_change", control=64, value=0, time=240)])
            path = Path(tmp)/"pedal.mid"
            midi.save(path)
            events, _ = midi_events(path)
            self.assertEqual(len(events), 1)
            self.assertAlmostEqual(events[0].duration, 1.5)
            clipped = clip_events(events, .5, .4)
            self.assertAlmostEqual(clipped[0].duration, .4)
            self.assertAlmostEqual(activity_seconds(clipped), .4)

    def test_polyphonic_voice_separation_preserves_notes(self):
        events = [NoteEvent("p", 60, 0., 1.), NoteEvent("p", 64, 0., 1.), NoteEvent("p", 67, .5, 1.)]
        bodies = list(abc_voices(events, duration=2.))
        self.assertEqual(len(bodies), 2)
        parsed = [e for body in bodies for e in abc_to_events("X:1\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\n"+body)]
        self.assertEqual(sorted(e.pitch for e in parsed), [60, 64, 67])
        self.assertEqual(sorted((e.pitch, e.onset, e.duration) for e in parsed), sorted((e.pitch, e.onset, e.duration) for e in events))

    def test_slakh_manifests_feed_training_and_preserve_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_slakh_fixture(root/"source")
            report = prepare_slakh(root/"source", root/"prepared")
            self.assertEqual(report["audio_examples"], 3)
            rows = read_training_manifest(root/"prepared/audio_train.jsonl", "audio-sft")
            self.assertEqual(len(rows), 1)
            wave, sr = sf.read(rows[0]["audio"])
            self.assertEqual(sr, 32000)
            self.assertEqual(wave.shape, (480000,))
            symbols = read_training_manifest(root/"prepared/symbolic_train.jsonl", "symbolic-sft")
            self.assertEqual(len(symbols), 2)
            self.assertTrue(all(abc_to_events(Path(r["abc"]).read_text()) for r in symbols))
            self.assertNotIn("reward_inputs", rows[0])
            with self.assertRaises(FileExistsError):
                prepare_slakh(root/"source", root/"prepared")

    def test_symbolic_ppo_needs_no_target_abc(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"train.jsonl"
            write_jsonl(path, [{"sample_id": "a", "split": "train", "prefix": "K:C\n"}])
            row = read_training_manifest(path, "symbolic-ppo")[0]
            payload, prompt = symbolic_example(row, Tokenizer(), "cpu", supervised=False)
            self.assertEqual(payload, {})
            self.assertEqual(tuple(prompt.shape), (1, 3))

    def test_empty_chords_and_bare_major_key_are_not_corrupted(self):
        from planning.blueprint_parser import MusicBlueprintParser
        parser = MusicBlueprintParser()
        self.assertEqual(parser._parse_chords([], "C"), [])
        self.assertEqual(parser._parse_chords(None, "C"), ["C"])
        self.assertEqual(parser._parse_chords(None, "F#min"), ["F#m"])

    def test_mixture_has_equal_source_counts_and_unique_draw_ids(self):
        from collections import Counter
        from training.prepare_dataset import mix_audio
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("audio", "blueprint", "events"):
                (root/name).write_text("fixture")
            base = {"split": "train", "audio": "audio", "blueprint": "blueprint", "symbolic_outputs": "events"}
            write_jsonl(root/"slakh.jsonl", [{**base, "sample_id": "s1"}])
            write_jsonl(root/"musicpro.jsonl", [{**base, "sample_id": "m1"}, {**base, "sample_id": "m2"}])
            mix_audio(root/"slakh.jsonl", root/"musicpro.jsonl", root/"mixed")
            rows = read_training_manifest(root/"mixed/audio_train.jsonl", "audio-sft")
            self.assertEqual(Counter(r["mixture_source"] for r in rows), {"slakh": 2, "musicpro": 2})
            self.assertEqual(len({r["sample_id"] for r in rows}), 4)

    def test_visual_training_excludes_held_out_movie_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"video.avi").write_bytes(b"file-hash-fixture")
            common = {"dataset": "test_fixture", "video": "video.avi", "movie_id": "heldout_movie"}
            write_jsonl(root/"heldout.jsonl", [{**common, "sample_id": "test_pair", "split": "test"}])
            write_jsonl(root/"labels.jsonl", [{**common, "sample_id": "visual_window", "split": "train",
                "prior_5d": [.2]*5, "label_source": "test", "label_protocol": "test", "start_time": 0., "end_time": 2.}])
            report = prepare_index(root/"labels.jsonl", root/"prepared", kind="cineparse", exclude_manifest=root/"heldout.jsonl")
            self.assertEqual(report["excluded_from_visual_training"], ["visual_window"])
            self.assertEqual((root/"prepared/train.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
