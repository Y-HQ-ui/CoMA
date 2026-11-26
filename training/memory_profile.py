"""Optional CUDA memory accounting for real training steps."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch


class TrainingMemoryProfile:
    def __init__(self, root, device, metadata):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("training memory profiling requires a CUDA device")
        self.path = Path(root) / "memory_profile.json"
        gpu = torch.cuda.get_device_properties(self.device)
        self.report = {
            "status": "running", "pid": os.getpid(),
            "hardware": {"name": gpu.name, "total_memory_bytes": gpu.total_memory,
                         "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "device": str(self.device), "torch": torch.__version__,
                         "cuda": torch.version.cuda},
            "protocol": {
                "unit": "GiB (2^30 bytes)", "batch_size": metadata["batch_size"],
                "stage": metadata["stage"], "base_model": metadata["base_model"]["path"],
                "trainable_parameters": metadata["trainable_parameters"],
                "reward_backend": metadata["reward_backend"],
                "scope": "resident policy, rollout, reward, old likelihood, backward and AdamW update",
                "peaks": "reset before each step; phase snapshots are cumulative within the step",
                "warmup": "step 0 includes fresh AdamW state initialization; later steps retain optimizer state",
            },
            "steps": [],
        }
        self.flush()

    def flush(self):
        self.path.write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def begin_step(self, step, sample_id):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.monotonic()
        self.current = {"step": step, "sample_id": sample_id, "phases": {}}
        self.report["steps"].append(self.current)
        self.flush()

    def mark(self, phase, **details):
        torch.cuda.synchronize(self.device)
        values = {"allocated_peak_bytes": torch.cuda.max_memory_allocated(self.device),
                  "reserved_peak_bytes": torch.cuda.max_memory_reserved(self.device),
                  "allocated_current_bytes": torch.cuda.memory_allocated(self.device),
                  "reserved_current_bytes": torch.cuda.memory_reserved(self.device)}
        values.update({key.removesuffix("_bytes") + "_gib": value / 1024**3
                       for key, value in list(values.items())})
        self.current["phases"][phase] = {
            **values, "elapsed_seconds": time.monotonic() - self.started, **details}
        self.flush()

    def finish(self):
        rows = [row["phases"]["after_update"] for row in self.report["steps"]]
        self.report["summary"] = {
            "steps": len(rows),
            "optimizer_updates": sum(row["optimizer_updates"] for row in rows),
            "allocated_peak_gib": max(row["allocated_peak_gib"] for row in rows),
            "reserved_peak_gib": max(row["reserved_peak_gib"] for row in rows),
        }
        self.report["status"] = "completed"
        self.flush()
