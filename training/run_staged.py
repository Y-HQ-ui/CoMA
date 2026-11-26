"""Execute Algorithm 1 with separate processes and immutable stage artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.provenance import sha256_file


def run(config_path, output_dir):
    path = Path(config_path).resolve()
    spec = json.loads(path.read_text())
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("staged training requires a new output directory")
    def resolve(key, required=True):
        value = spec.get(key)
        if not value:
            if required:
                raise ValueError(f"missing staged configuration: {key}")
            return ""
        p = Path(value)
        p = p if p.is_absolute() else path.parent/p
        if not p.is_file():
            raise FileNotFoundError(p)
        return str(p.resolve())
    symbolic_manifest, audio_manifest = resolve("symbolic_manifest"), resolve("audio_manifest")
    symbolic_ppo_manifest = resolve("symbolic_ppo_manifest", False) or symbolic_manifest
    guidance_manifest = resolve("guidance_manifest", False) or audio_manifest
    paper_config = resolve("paper_config")
    starts = {family: resolve(f"{family}_sft_checkpoint", False) for family in ("symbolic", "audio")}
    device = spec.get("device", "cpu")
    seed = int(spec.get("seed", 123))
    root.mkdir(parents=True, exist_ok=True)
    records = []
    result = {"algorithm": "paper Algorithm 1", "status": "running", "config_sha256": sha256_file(path), "spec": spec,
              "stages": records, "scope": "single phrase per symbolic manifest row; fixed generated symbolic guidance in audio PPO"}
    def execute(name, command):
        entry = {"stage": name, "command": command, "status": "running"}
        records.append(entry)
        (root/"staged_protocol.json").write_text(json.dumps(result, indent=2))
        with (root/f"{name}.log").open("w") as log:
            process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=process.returncode, status="completed" if process.returncode == 0 else "failed")
        if process.returncode:
            result["status"] = "failed"
        (root/"staged_protocol.json").write_text(json.dumps(result, indent=2))
        if process.returncode:
            raise RuntimeError(f"stage {name} failed; see {root/name}.log")
    def stage(name, manifest, warm=""):
        settings = spec.get("stages", {}).get(name, {})
        extra = {"ppo_epochs", "ppo_clip", "value_clip", "gae_lambda", "entropy_coef", "value_coef",
                 "max_grad_norm", "target_kl", "discount_per_unit", "discount_unit_seconds", "audio_window_seconds"}
        if set(settings)-extra-{"steps", "epochs", "learning_rate"}:
            raise ValueError(f"unknown stage settings for {name}: {sorted(set(settings)-extra-{'steps', 'epochs', 'learning_rate'})}")
        command = [sys.executable, "-m", "training.run", "--stage", name, "--manifest", manifest,
            "--output-dir", str(root/name), "--device", device, "--seed", str(seed),
            "--steps", str(settings.get("steps", 100)), "--epochs", str(settings.get("epochs", 5)),
            "--learning-rate", str(settings.get("learning_rate", 1e-5)),
            "--max-new-tokens", str(spec.get("max_new_tokens", 256)), "--paper-config", paper_config]
        for key in sorted(extra & set(settings)):
            command += ["--"+key.replace("_", "-"), str(settings[key])]
        if warm:
            command += ["--warm-start-checkpoint", warm]
        if spec.get("disable_consistency"):
            command += ["--disable-consistency"]
        if spec.get("skip_weight_hash"):
            command += ["--skip-weight-hash"]
        execute(name, command)
        return str(root/name/"policy.pt")
    for family, manifest in (("symbolic", symbolic_manifest), ("audio", audio_manifest)):
        if starts[family]:
            records.append({"stage": family+"-sft", "status": "provided_checkpoint", "path": starts[family],
                            "sha256": sha256_file(starts[family])})
        else:
            starts[family] = stage(family+"-sft", manifest)
    symbolic = stage("symbolic-ppo", symbolic_ppo_manifest, starts["symbolic"])
    execute("frozen_guidance", [sys.executable, "-m", "training.prepare_guidance", "--manifest", guidance_manifest,
        "--symbolic-checkpoint", symbolic, "--output-dir", str(root/"guidance"), "--device", device,
        "--seed", str(seed), "--max-new-tokens", str(spec.get("guidance_max_new_tokens", 384))])
    stage("audio-ppo", str(root/"guidance/manifest.jsonl"), starts["audio"])
    result["status"] = "completed"
    (root/"staged_protocol.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "stages": len(result["stages"])}))


if __name__ == "__main__":
    main()
