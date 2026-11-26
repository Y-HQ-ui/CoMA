"""Audit actual paper-objective PPO artifacts without re-running generation."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from experiments.provenance import sha256_file


def verify(run_dir, warm_start):
    root, warm_start = Path(run_dir).resolve(), Path(warm_start).resolve()
    protocol = json.loads((root/"training_protocol.json").read_text())
    completion = json.loads((root/"completion.json").read_text())
    trace = json.loads((root/"reward_trace.json").read_text())
    logs = [json.loads(line) for line in (root/"training.jsonl").read_text().splitlines() if line.strip()]
    if completion["status"] != "completed" or completion["steps"] != len(logs) or len(trace) != len(logs):
        raise ValueError("incomplete run or reward/log denominator mismatch")
    if protocol["reward_backend"] != "paper":
        raise ValueError("this audit requires actual paper-objective training")
    if [row["step"] for row in logs] != list(range(len(logs))):
        raise ValueError("unexpected missing/duplicated update steps")
    old = torch.load(warm_start, map_location="cpu", weights_only=True)
    new = torch.load(root/"policy.pt", map_location="cpu", weights_only=True)
    if old["metadata"]["stage"] != protocol["family"]+"-sft":
        raise ValueError("comparison requires the actual supervised warm start")
    if new["metadata"] != protocol or set(old["parameters"]) != set(new["parameters"]):
        raise ValueError("checkpoint/protocol/parameter identity mismatch")
    if not all(torch.isfinite(tensor).all() for tensor in new["parameters"].values()):
        raise ValueError("nonfinite trained parameter")
    changed = [name for name in old["parameters"] if not torch.equal(old["parameters"][name], new["parameters"][name])]
    if not any("value" not in name for name in changed):
        raise ValueError("no actual generator adapter parameter changed")
    counts, unavailable = Counter(), Counter()
    mass_errors = []
    for index, row in enumerate(trace):
        if row["step"] != index or row["sample_id"] != logs[index]["sample_id"]:
            raise ValueError("reward trace identity mismatch")
        if "aggregation" not in row:
            if row.get("status") not in {"invalid_score", "no_notes"} or row.get("reward") != 0:
                raise ValueError("invalid score must be retained with zero reward")
            counts[row["status"]] += 1
            continue
        counts["measured"] += 1
        aggregate = row["aggregation"]
        measured = 0.
        for name, weight in aggregate["weights"].items():
            item = row["components"][name]
            if item["status"] == "ok":
                if not np.isfinite([item["raw"], item["reward"]]).all() or not 0 <= item["reward"] <= 1:
                    raise ValueError("invalid component scale")
                measured += weight*item["reward"]
            else:
                unavailable[name] += 1
                if weight and name not in aggregate["zeroed_unavailable"]:
                    raise ValueError("unavailable component was not disclosed")
        if not np.isclose(measured, aggregate["reward"], atol=1e-10):
            raise ValueError("weighted reward differs from components")
        if "windows" in row:
            if row["evidence"]["audio_sha256"] != sha256_file(root/f"sample_{index:05d}.wav"):
                raise ValueError("reward does not belong to saved rollout audio")
            error = abs(sum(w["reward_mass"] for w in row["windows"])-row["assigned_reward_mass"])
            if error > 1e-6:
                raise ValueError("window reward mass was not conserved")
            mass_errors.append(error)
    trajectory = json.loads((root/"weight_trajectory.json").read_text())
    weights = trajectory["rows"]
    if protocol["family"] == "audio" and len(weights) != len(logs):
        raise ValueError("missing audio coefficient step")
    config = trajectory["protocol"]["weights"]
    for row in weights:
        names = list(trajectory["protocol"]["active_components"])
        current = np.array([row["weights"][name] for name in names])
        before = np.array([row["before"][name] for name in names])
        delta = float(np.linalg.norm(current-before))
        if not np.isfinite(current).all() or not np.isclose(current.sum(), 1., atol=1e-10) or (current < config["floor"]-1e-10).any():
            raise ValueError("coefficient simplex/floor violated")
        if delta > config["max_step"]+1e-10 or not np.isclose(delta, row["l2_step"], atol=1e-10):
            raise ValueError("coefficient step bound/trace mismatch")
    return {"status": "software_artifacts_verified", "run": str(root), "steps": len(logs),
            "actual_optimizer_updates": sum(len(row["stats"]["updates"]) for row in logs),
            "reward_samples": dict(counts), "unavailable_active_components": dict(unavailable),
            "changed_trainable_tensors": len(changed),
            "changed_generator_tensors": sum("value" not in name for name in changed),
            "checkpoint_sha256": sha256_file(root/"policy.pt"), "warm_start_sha256": sha256_file(warm_start),
            "weights_checked": len(weights), "max_reward_mass_error": max(mass_errors, default=None),
            "paper_runtime": protocol["paper_runtime"], "discount": protocol["discount"],
            "source_recorded_at_training": protocol.get("source"),
            "limitation": "few-step artifact/parameter/credit verification; does not establish generation quality or original-paper reproduction"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--warm-start", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = verify(args.run_dir, args.warm_start)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("choose a new verification output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("status", "steps", "reward_samples", "changed_generator_tensors")}))


if __name__ == "__main__":
    main()
