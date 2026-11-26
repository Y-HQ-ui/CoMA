"""Algorithm 1 line 10: generate conditions under a frozen symbolic checkpoint.

This writes new manifests and provenance, never substitutes target MIDI/audio
for a generated score, and never modifies the symbolic checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from experiments.provenance import sha256_file, source_fingerprint
from audio.symbolic_renderer import generate_blueprint_score
from planning import load_blueprint_json
from planning.timeline import merge_overlapping_segments
from training.data import read_training_manifest


def prepare(manifest, checkpoint, output_dir, *, model_dir="", device="cpu", seed=123, max_new_tokens=384):
    import torch
    rows = read_training_manifest(manifest, "guidance")
    checkpoint = Path(checkpoint).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved.get("metadata", {})
    if metadata.get("family") != "symbolic" or metadata.get("stage") not in {"symbolic-sft", "symbolic-ppo"}:
        raise ValueError("guidance requires a trained symbolic policy checkpoint")
    del saved
    digest = sha256_file(checkpoint)
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("use a new guidance directory")
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"kind": "frozen_policy_generation", "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                "input_manifest_sha256": sha256_file(manifest), "seed": seed, "max_new_tokens": max_new_tokens,
                "condition": "blueprint only; target audio never read by score generation",
                "source": source_fingerprint(Path(__file__).resolve().parents[1]), "records": []}
    output_rows = []
    for index, row in enumerate(rows):
        blueprint = merge_overlapping_segments(load_blueprint_json(row["blueprint"]))
        sample_root = root/f"sample_{index:06d}"
        symbolic = generate_blueprint_score(blueprint, output_dir=str(sample_root), model_dir=model_dir,
            device=device, max_new_tokens=max_new_tokens, seed=seed+index, policy_checkpoint=str(checkpoint))
        events_path = sample_root/"symbolic_outputs.json"
        events_path.write_text(json.dumps(symbolic, indent=2))
        provenance = {"kind": "frozen_policy_generation", "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                      "events_sha256": sha256_file(events_path), "blueprint_sha256": sha256_file(row["blueprint"]),
                      "seed": seed+index, "sample_id": row["sample_id"]}
        output_rows.append({**row, "symbolic_outputs": str(events_path), "symbolic_guidance": provenance})
        protocol["records"].append(provenance)
        (root/"manifest.jsonl").write_text("\n".join(json.dumps(r) for r in output_rows)+"\n")
        (root/"guidance_protocol.json").write_text(json.dumps(protocol, indent=2))
    if sha256_file(checkpoint) != digest:
        raise RuntimeError("symbolic checkpoint changed during frozen-guidance preparation")
    protocol.update(status="completed", sample_count=len(rows), checkpoint_unchanged=True)
    (root/"guidance_protocol.json").write_text(json.dumps(protocol, indent=2))
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--symbolic-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-dir", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-new-tokens", type=int, default=384)
    args = p.parse_args()
    result = prepare(args.manifest, args.symbolic_checkpoint, args.output_dir, model_dir=args.model_dir,
                     device=args.device, seed=args.seed, max_new_tokens=args.max_new_tokens)
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"]}))


if __name__ == "__main__":
    main()
