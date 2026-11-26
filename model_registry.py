"""Offline model discovery. Read metadata only; never silently downloads weights."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def inspect_checkpoint(path):
    path = Path(path).resolve()
    config_path = path/"config.json"
    if not config_path.is_file():
        return {"path":str(path),"status":"missing_config"}
    try:
        config = json.loads(config_path.read_text())
    except (ValueError, OSError) as exc:
        return {"path":str(path),"status":"invalid_config","reason":str(exc)}
    kind = config.get("model_type", "unknown")
    family = "musicgen" if kind in {"musicgen","musicgen_melody"} else "mupt" if "mupt" in str(path).lower() and kind == "llama" else kind
    files = []
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path/index_name
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values())); break
    if not files:
        files = [n for n in ("model.safetensors","pytorch_model.bin") if (path/n).exists()][:1]
    missing = [n for n in files if not (path/n).is_file() or (path/n).stat().st_size < 1024]
    tokenizer_present = any((path/n).exists() for n in ("tokenizer.json","tokenizer.model","spiece.model")) or all((path/n).exists() for n in ("vocab.json","merges.txt","tokenizer_config.json"))
    status = "ready" if files and not missing and tokenizer_present else "incomplete"
    return {"path":str(path),"family":family,"model_type":kind,"status":status,
            "weight_files":files,"missing_files":missing,"tokenizer_present":tokenizer_present,
            "weight_bytes":sum((path/n).stat().st_size for n in files if (path/n).is_file()),
            "hidden_size":config.get("hidden_size",config.get("decoder",{}).get("hidden_size")),
            "verification":"metadata_and_file_presence_only"}


def discover_models(roots=None):
    roots = roots or [ROOT/"models"]
    found = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir(): continue
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {".git",".cache","node_modules","experiments","reports","outputs","output"}]
            if len(Path(base).relative_to(root).parts) >= 3: dirs[:] = []
            if "config.json" in files:
                info = inspect_checkpoint(base)
                if info.get("family") in {"mupt","musicgen"}: found[info["path"]] = info
    return sorted(found.values(),key=lambda item:(item["family"],item["weight_bytes"],item["path"]))


def resolve_model(family, explicit=None):
    requested = explicit or os.getenv(f"MUSICAGENT_{family.upper()}_MODEL_DIR")
    if requested:
        info = inspect_checkpoint(requested)
        if info.get("family") != family or info["status"] != "ready":
            raise ValueError(f"Invalid local {family} checkpoint: {info}")
        if family=="musicgen" and info["model_type"]!="musicgen":
            raise ValueError("this renderer supports MusicgenForConditionalGeneration; MusicGen Melody requires a separate backend")
        return info["path"]
    choices = [m for m in discover_models() if m["family"] == family and m["status"] == "ready"]
    if family == "musicgen": choices = [m for m in choices if m["model_type"] == "musicgen"]
    if not choices:
        raise FileNotFoundError(f"No complete local {family} checkpoint; configure MUSICAGENT_{family.upper()}_MODEL_DIR")
    return choices[0]["path"]


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",action="append"); parser.add_argument("--output")
    args=parser.parse_args()
    report={"models":discover_models(args.root),"automatic_selection":"smallest complete checkpoint; no weight conversion/download"}
    content=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(content)
    print(content)
