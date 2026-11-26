"""One-source qualitative evidence with time/frequency axes and shared dB scale."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft
from experiments.provenance import sha256_file


def render(spec_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reward.evidence import extract_audio_features
    from planning import load_blueprint_json
    path = Path(spec_path).resolve()
    spec = json.loads(path.read_text())
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("use a new qualitative output directory")
    def resolve(value):
        value = Path(value)
        return value if value.is_absolute() else path.parent/value
    signals = []
    for item in spec["systems"]:
        filename = resolve(item["audio"])
        wave, sr = sf.read(filename, always_2d=True)
        if not np.isfinite(wave).all() or not len(wave):
            raise ValueError("invalid waveform")
        signals.append((item["name"], filename, wave.mean(1), sr))
    if not signals or len({(len(w), sr) for _, _, w, sr in signals}) != 1:
        raise ValueError("all displayed systems must have the same sample rate and duration")
    if len({name for name, *_ in signals}) != len(signals):
        raise ValueError("duplicate system label")
    duration = len(signals[0][2])/signals[0][3]
    priors = json.loads(resolve(spec["prior"]).read_text())
    visual = priors["frame_features"]
    blueprint = load_blueprint_json(str(resolve(spec["blueprint"])))
    event_file = resolve(spec["symbolic_outputs"])
    events = [event for record in json.loads(event_file.read_text()) for event in record["events"]]
    root.mkdir(parents=True, exist_ok=True)
    rows = 3+2*len(signals)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2*rows), sharex=True, layout="constrained")
    axes[0].plot(visual["times"], visual["motion_intensity"], ".-", label="measured motion")
    for timestamp, cut in zip(visual["times"], visual["cut_flags"]):
        if cut:
            axes[0].axvline(timestamp, color="tab:red", alpha=.6)
    axes[0].set(ylabel="Visual [0,1]", title=spec.get("title", "Qualitative evidence"));axes[0].legend()
    for segment in blueprint.segments:
        axes[1].plot([segment.start_time, segment.end_time], [segment.tempo_bpm]*2, color="tab:blue")
        axes[1].axvline(segment.start_time, color="gray", alpha=.2)
        axes[1].text((segment.start_time+segment.end_time)/2, segment.tempo_bpm+1, segment.section,
                     ha="center", fontsize=8)
    axes[1].set(ylabel="Blueprint BPM")
    tracks = sorted({e["track_id"] for e in events})
    for index, track in enumerate(tracks):
        for note_index, event in enumerate([e for e in events if e["track_id"] == track]):
            axes[2].plot([event["onset"], event["onset"]+event["duration"]], [event["pitch"]]*2,
                         linewidth=2, color=f"C{index%10}", label=track if note_index == 0 else None)
    axes[2].set(ylabel="Score MIDI pitch", title="Declared symbolic score, not a transcription of baseline audio")
    axes[2].legend(loc="upper right", fontsize=8)
    dB_floor, dB_ceiling = -80., 0.
    evidence = []
    for index, (name, filename, wave, sr) in enumerate(signals):
        n_fft = round(.032*sr)
        hop = round(.01*sr)
        frequencies, times, spectrum = stft(wave, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft-hop,
                                             boundary=None, padded=False)
        # Shared full-scale reference: no per-system peak normalization.
        db = 20*np.log10(np.maximum(np.abs(spectrum), 1e-4))
        ax = axes[3+2*index]
        image = ax.pcolormesh(times, frequencies, db, shading="auto", vmin=dB_floor, vmax=dB_ceiling, cmap="magma")
        ax.set(ylabel="Frequency (Hz)", title=f"{name}: mono STFT, full-scale reference", ylim=(0, min(8000, sr/2)))
        fig.colorbar(image, ax=ax, label="dB re full-scale 1")
        feature = extract_audio_features(str(filename))
        ax = axes[4+2*index]
        ax.plot(feature["frame_times"], feature["onset"], label="normalized spectral flux")
        ax.plot(feature["frame_times"], feature["rms_absolute"], label="absolute RMS")
        ax.set(ylabel="Onset / RMS");ax.legend(fontsize=8, loc="upper right")
        for interval in spec.get("repair_masks", []):
            start, end = interval
            if not 0 <= start < end <= duration:
                raise ValueError("invalid repair mask")
            ax.axvspan(start, end, color="tab:green", alpha=.12)
        evidence.append({"system": name, "path": str(filename.resolve()), "sha256": sha256_file(filename)})
    axes[-1].set(xlabel="Time (seconds)", xlim=(0, duration))
    fig.savefig(root/"case.png", dpi=140);fig.savefig(root/"case.pdf");plt.close(fig)
    protocol = {"systems": evidence, "sample_rate": signals[0][3], "duration_seconds": duration,
                "stft": {"window_ms": 32, "hop_ms": 10, "window": "Hann", "full_scale": 1.,
                         "dB_limits": [dB_floor, dB_ceiling], "per_system_normalization": False},
                "symbolic_source_sha256": sha256_file(event_file), "spec": spec,
                "claim_scope": "single-case visualization of measured signals and declared controls; no source-separation or quality superiority inference"}
    (root/"case_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True);p.add_argument("--output-dir", required=True)
    args = p.parse_args();render(args.spec, args.output_dir)


if __name__ == "__main__":
    main()
