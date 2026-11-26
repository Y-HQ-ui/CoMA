"""Build explicitly synthetic fixtures and exercise all three dataset preparers."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from symbolic.events import NoteEvent, abc_to_events, export_midi
from training.prepare_dataset import (new_directory, prepare_slakh, prepare_index,
                                      write_json, write_jsonl, SPLITS)
from training.data import read_training_manifest


def make_slakh_fixture(root, duration=15.):
    root = Path(root)
    for index, split in enumerate(SPLITS):
        track = root/split/f"Track{index:04d}"
        track.mkdir(parents=True)
        events = []
        for i, start in enumerate(np.arange(0., duration, .5)):
            events.append(NoteEvent("piano", 60+index+i % 5, float(start), min(.45, duration-start), .6, 0, "melody"))
            events.append(NoteEvent("bass", 36+index+i % 3, float(start), min(.5, duration-start), .4, 32, "bass"))
        export_midi(events, track/"all_src.mid")
        sr = 32000
        t = np.arange(round(duration*sr))/sr
        wave = np.zeros_like(t)
        for event in events:
            age = t-event.onset
            active = (age >= 0) & (age < event.duration)
            wave += active*.08*event.velocity*np.sin(2*np.pi*(440*2**((event.pitch-69)/12))*age)
        sf.write(track/"mix.wav", wave, sr)


def verify(output):
    import cv2
    root = new_directory(output)
    make_slakh_fixture(root/"synthetic_slakh")
    slakh = prepare_slakh(root/"synthetic_slakh", root/"prepared_slakh")
    symbolic = read_training_manifest(root/"prepared_slakh/symbolic_train.jsonl", "symbolic-sft")
    audio = read_training_manifest(root/"prepared_slakh/audio_train.jsonl", "audio-sft")
    for row in symbolic:
        if not abc_to_events(Path(row["abc"]).read_text()):
            raise AssertionError("prepared symbolic example has no notes")
    musicpro, visual = [], []
    for index, split in enumerate(SPLITS):
        video = root/f"synthetic_{index}.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 64))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MJPG encoder unavailable")
        for frame in range(150):
            image = np.full((64, 64, 3), (index*40+frame) % 220, dtype=np.uint8)
            cv2.rectangle(image, (frame % 40, 8), (frame % 40+20, 40), (230, 20+index*30, 80), -1)
            writer.write(image)
        writer.release()
        prepared = root/"prepared_slakh"/f"audio_{split}.jsonl"
        item = json.loads(prepared.read_text().splitlines()[0])
        symbol = json.loads((root/"prepared_slakh"/f"symbolic_{split}.jsonl").read_text().splitlines()[0])
        base = {"sample_id": f"synthetic_video_{index}", "dataset": "synthetic_software_fixture_not_MusicPro",
                "split": split, "movie_id": f"synthetic_movie_{index}", "music_work_id": f"synthetic_work_{index}",
                "video": str(video), "audio": item["audio"], "blueprint": item["blueprint"],
                "symbolic_outputs": item["symbolic_outputs"], "input_condition": "V",
                "prefix": symbol["prefix"], "duration_seconds": 15.}
        musicpro.append(base)
        visual.append({**base, "start_time": 0., "end_time": 2., "prior_5d": [.2, .3, .4, .5, .6],
                       "label_source": "synthetic_test_labels", "label_protocol": "synthetic_fixture_v1", "hit_times": [.5]})
    write_jsonl(root/"musicpro_index.jsonl", musicpro)
    write_jsonl(root/"visual_index.jsonl", visual)
    pair_report = prepare_index(root/"musicpro_index.jsonl", root/"prepared_musicpro", kind="musicpro", extract_visual=True)
    visual_report = prepare_index(root/"visual_index.jsonl", root/"prepared_visual", kind="cineparse",
                                  exclude_manifest=root/"prepared_musicpro/all.jsonl")
    from vedio.train_cineparse import load_manifests
    parsed = load_manifests(root/"prepared_visual/train.jsonl", root/"prepared_visual/validation.jsonl")
    guidance = read_training_manifest(root/"prepared_musicpro/guidance_train.jsonl", "guidance")
    if "audio" in guidance[0]:
        raise AssertionError("guidance manifest exposes target audio")
    report = {"status": "passed", "scope": "synthetic software fixtures only; no paper data or performance claims",
              "slakh_audio_examples": slakh["audio_examples"], "slakh_symbolic_examples": slakh["symbolic_examples"],
              "train_abc_parsed": len(symbolic), "train_audio_examples": len(audio),
              "musicpro_audit": pair_report["audit"], "visual_counts": {key: len(value) for key, value in parsed.items()},
              "guidance_has_no_target_audio": True, "visual_report": visual_report}
    write_json(root/"verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
