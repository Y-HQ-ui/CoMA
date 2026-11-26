"""Measure real generation and refinement workloads on one or more GPUs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import traceback

import numpy as np
import torch

from audio.musicgen_renderer import MusicGenRendererSession
from experiments.provenance import sha256_file
from experiments.realization import RealizationBackend
from model_registry import inspect_checkpoint, resolve_model
from planning import load_blueprint_json
from reward.critic import diagnose_failure
from reward.refinement import GenerationBudget, refine_inference


class ProcessMemorySampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            rows = []
            elapsed = time.monotonic() - self.started
            for row in csv.reader(result.stdout.splitlines()):
                if len(row) == 3 and int(row[0].strip()) == os.getpid():
                    rows.append({"elapsed_seconds": elapsed,
                                 "gpu_uuid": row[1].strip(),
                                 "used_bytes": int(row[2].strip()) * 1024**2})
            with self.lock:
                self.samples.extend(rows)
        except Exception as exc:
            with self.lock:
                self.errors.append(str(exc))

    def run(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self.sample()
        self.thread.start()

    def snapshot(self):
        with self.lock:
            samples = list(self.samples)
            errors = list(self.errors)
        per_gpu = {}
        totals = {}
        for row in samples:
            uuid = row["gpu_uuid"]
            per_gpu[uuid] = max(per_gpu.get(uuid, 0), row["used_bytes"])
            elapsed = row["elapsed_seconds"]
            totals[elapsed] = totals.get(elapsed, 0) + row["used_bytes"]
        return {"peak_bytes": max(per_gpu.values(), default=None),
                "per_gpu_peak_bytes": per_gpu,
                "total_peak_bytes": max(totals.values(), default=None),
                "sample_count": len(samples), "errors": errors,
                "gpu_uuids": sorted({r["gpu_uuid"] for r in samples})}

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError("process memory sampler did not stop")
        self.sample()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def snapshot(devices, sampler, started):
    for device in devices:
        torch.cuda.synchronize(device)
    result = {"elapsed_seconds": time.monotonic() - started,
              "devices": {}, "process_memory": sampler.snapshot()}
    for device in devices:
        memory = {"allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                  "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
                  "allocated_current_bytes": torch.cuda.memory_allocated(device),
                  "reserved_current_bytes": torch.cuda.memory_reserved(device)}
        for name in ("allocated_peak", "reserved_peak", "allocated_current", "reserved_current"):
            memory[name + "_gib"] = memory[name + "_bytes"] / 1024**3
        result["devices"][device] = memory
    total_peak = result["process_memory"]["total_peak_bytes"]
    result["process_total_peak_gib"] = total_peak / 1024**3 if total_peak is not None else None
    if len(devices) == 1:
        result.update(result["devices"][devices[0]])
        result["process_peak_gib"] = result["process_total_peak_gib"]
    else:
        # Separate device peaks can occur at different times.
        for name in ("allocated_peak", "reserved_peak"):
            result[name + "_sum_of_device_peaks_gib"] = sum(
                memory[name + "_gib"] for memory in result["devices"].values())
    return result


def check_audio(path):
    import soundfile as sf
    audio, rate = sf.read(path, always_2d=True)
    if not np.isfinite(audio).all() or not len(audio):
        raise ValueError("invalid generated waveform")
    return {"path": str(path), "sha256": sha256_file(path), "sample_rate": rate,
            "samples": len(audio), "channels": audio.shape[1],
            "duration_seconds": len(audio) / rate,
            "rms": float(np.sqrt(np.mean(audio**2))),
            "peak_amplitude": float(np.abs(audio).max())}


def summarize(records):
    def statistics_for(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return {"mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "max": max(values)} if values else None

    groups = {}
    for row in records:
        if row["status"] != "ok":
            continue
        for phase in ("one_shot", "with_refinement"):
            if phase in row:
                groups.setdefault(phase, []).append(row[phase])
    result = {}
    for phase, rows in groups.items():
        result[phase] = {"trials": len(rows)}
        for key in ("allocated_peak_gib", "reserved_peak_gib", "process_peak_gib", "elapsed_seconds",
                    "process_total_peak_gib", "allocated_peak_sum_of_device_peaks_gib",
                    "reserved_peak_sum_of_device_peaks_gib"):
            if any(key in row for row in rows):
                result[phase][key] = statistics_for(rows, key)
        result[phase]["devices"] = {
            device: {key: statistics_for([row["devices"][device] for row in rows], key)
                     for key in ("allocated_peak_gib", "reserved_peak_gib")}
            for device in rows[0].get("devices", {})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--musicgen-model-dir", required=True)
    parser.add_argument("--mode", choices=("musicgen", "coma"), default="coma")
    parser.add_argument("--mupt-model-dir", default="")
    parser.add_argument("--mupt-device", default="cuda:0")
    parser.add_argument("--musicgen-device", default="cuda:0")
    parser.add_argument("--symbolic-policy-checkpoint", default="")
    parser.add_argument("--adapter-checkpoint", default="")
    parser.add_argument("--blueprint", default="examples/blueprint_smoke.json")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--warmup-seed", type=int, default=1900)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--refinements", type=int, default=5)
    parser.add_argument("--symbolic-max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    devices = sorted({args.musicgen_device} | ({args.mupt_device} if args.mode == "coma" else set()))
    for device in devices:
        parsed = torch.device(device)
        if parsed.type != "cuda" or parsed.index is None or parsed.index >= torch.cuda.device_count():
            raise ValueError(f"use an available explicit CUDA device, got {device}")
    if args.warmups < 0 or args.refinements < 0 or not 0 < args.duration <= 30:
        raise ValueError("invalid warmup, refinement or duration setting")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.musicgen_device)
    torch.set_num_threads(1)
    audio_base = resolve_model("musicgen", args.musicgen_model_dir)
    gpu = torch.cuda.get_device_properties(args.musicgen_device)
    blueprint_data = json.loads(Path(args.blueprint).read_text())
    if len(blueprint_data["segments"]) != 1:
        raise ValueError("duration override requires a single-segment blueprint")
    blueprint_data["segments"][0].update(start_time=0.0, end_time=args.duration)
    blueprint_data["source"] = "memory_profile_synthetic_two_track_fixture"
    write_json(root / "input_blueprint.json", blueprint_data)
    blueprint = load_blueprint_json(str(root / "input_blueprint.json"))
    report = {
        "status": "loading", "args": vars(args), "pid": os.getpid(),
        "hardware": {"name": gpu.name, "total_memory_bytes": gpu.total_memory,
                     "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "devices": {device: {"name": torch.cuda.get_device_properties(device).name,
                                          "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory}
                                 for device in devices},
                     "torch": torch.__version__, "cuda": torch.version.cuda},
        "protocol": {"batch_size": 1, "generator_dtype": "float16", "device_count": len(devices),
                     "device_mapping": {"musicgen": args.musicgen_device,
                                        "mupt": args.mupt_device if args.mode == "coma" else None},
                     "sample_rate": 32000, "duration_seconds": args.duration,
                     "input": "fixed synthetic 15-second two-track blueprint" if args.mode == "coma" else "fixed text prompt",
                     "scope": "resident generators, generation, serialization and artifact scoring; no online video encoder or LLM",
                     "adapter": args.adapter_checkpoint or None,
                     "scoring": "artifact_proxy_v1" if args.mode == "coma" else None,
                     "replanning": "deterministic rules" if args.mode == "coma" else None,
                     "process_memory_sampling": "nvidia-smi current PID, 100ms wait between samples",
                     "multi_gpu_memory": "per-device PyTorch peaks; sum of device peaks is an upper bound; process total is summed within each nvidia-smi query",
                     "memory_unit": "GiB (2^30 bytes)",
                     "warmup_workload": "one-shot generation with a fixed seed",
                     "refinement_rule": "fixed proposal count; gain > 1e-4; no threshold/patience early stop",
                     "audio_sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 250, "guidance_scale": 3.0}},
        "models": {"musicgen": inspect_checkpoint(audio_base)}, "warmups": [], "records": [],
    }
    report_path = root / "memory_report.json"
    write_json(report_path, report)
    loading_started = time.monotonic()
    symbolic_base = ""
    if args.mode == "coma":
        from track.melody.infer import load_model
        symbolic_base = resolve_model("mupt", args.mupt_model_dir)
        load_model(symbolic_base, args.mupt_device, policy_checkpoint=args.symbolic_policy_checkpoint)
        report["models"]["mupt"] = inspect_checkpoint(symbolic_base)
    renderer = MusicGenRendererSession.get(model_dir=audio_base, device=args.musicgen_device)
    report["models"]["musicgen"].update(
        decoder_layers=renderer.model.config.decoder.num_hidden_layers,
        parameter_count=sum(p.numel() for p in renderer.model.parameters()),
        config_sha256=sha256_file(Path(audio_base) / "config.json"),
    )
    for field in ("adapter_checkpoint", "symbolic_policy_checkpoint"):
        if getattr(args, field):
            report["models"][field] = {"path": str(Path(getattr(args, field)).resolve()),
                                      "sha256": sha256_file(getattr(args, field))}
    for device in devices:
        torch.cuda.synchronize(device)
    report["model_load_seconds"] = time.monotonic() - loading_started

    def generate(run_dir, seed):
        if args.mode == "musicgen":
            run_dir.mkdir(parents=True, exist_ok=False)
            path = renderer.render_one(
                "Instrumental cinematic piano and acoustic bass, calm introduction, C major, 120 BPM, clear melody.",
                output_path=str(run_dir / "soundtrack.wav"), seed=seed, duration=args.duration,
            )
            return path, None, None, None
        backend = RealizationBackend(str(run_dir), symbolic_base, audio_base,
                                     args.mupt_device, args.musicgen_device,
                                     args.adapter_checkpoint,
                                     symbolic_policy_checkpoint=args.symbolic_policy_checkpoint,
                                     max_new_tokens=args.symbolic_max_new_tokens)
        budget = GenerationBudget(256)
        initial = backend.generate(blueprint, seed=seed, budget=budget)
        return initial.audio_manifest["soundtrack_path"], initial, backend, budget

    for index in range(args.warmups):
        print(f"warmup {index + 1}/{args.warmups}", flush=True)
        started = time.monotonic()
        path, initial, backend, budget = generate(root / f"warmup_{index}", args.warmup_seed)
        report["warmups"].append({"index": index, "audio": check_audio(path),
                                   "elapsed_seconds": time.monotonic() - started})
        del initial, backend, budget
        write_json(report_path, report)

    report["status"] = "measuring"
    for seed in args.seeds:
        print(f"measuring seed {seed}", flush=True)
        for device in devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        sampler = ProcessMemorySampler()
        sampler.start()
        started = time.monotonic()
        row = {"seed": seed, "status": "running"}
        report["records"].append(row)
        try:
            path, initial, backend, budget = generate(root / f"seed_{seed}", seed)
            row["one_shot"] = snapshot(devices, sampler, started)
            row["one_shot"]["audio"] = check_audio(path)
            row["one_shot"]["generation_calls"] = budget.calls if budget else 1
            write_json(report_path, report)
            print(json.dumps({"seed": seed, "one_shot": row["one_shot"]}), flush=True)
            if initial is not None and args.refinements:
                def fixed_budget_diagnosis(current):
                    diagnosis = diagnose_failure(current.rewards, global_threshold=.55, component_threshold=.45)
                    diagnosis["repair_required"] = True
                    return diagnosis
                final, trace = refine_inference(
                    initial, backend.regenerate, budget=budget, max_iterations=args.refinements,
                    patience=args.refinements + 1, seed=seed,
                    decision_callback=fixed_budget_diagnosis,
                )
                row["with_refinement"] = snapshot(devices, sampler, started)
                row["with_refinement"]["audio"] = check_audio(final.audio_manifest["soundtrack_path"])
                row["with_refinement"]["trace"] = trace
                if len(trace["history"]) != args.refinements or any(
                        "error" in item or item.get("incomplete") for item in trace["history"]):
                    raise RuntimeError("refinement did not complete the declared number of evaluated proposals")
            row["status"] = "ok"
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
            row["failure_memory"] = snapshot(devices, sampler, started)
            print(row["error"], flush=True)
        finally:
            sampler.stop()
            write_json(root / f"seed_{seed}_process_memory.json", sampler.samples)
            row["process_sampler_errors"] = sampler.errors
            report["summary"] = summarize(report["records"])
            write_json(report_path, report)
        print(json.dumps({"seed": seed, "status": row["status"], "summary": report["summary"]}), flush=True)
    report["status"] = "completed" if all(r["status"] == "ok" for r in report["records"]) else "failed"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "summary": report["summary"]}), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
