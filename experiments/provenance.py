"""Canonical experiment IDs and artifact provenance, independent of Git availability."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def source_fingerprint(root):
    root=Path(root)
    files={str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob("*.py"))
           if not any(part in {"models","output",".venv","__pycache__"} for part in p.relative_to(root).parts)}
    return {"sha256":canonical_hash(files),"files":files}


def checkpoint_fingerprint(path,hash_weights=True):
    from model_registry import inspect_checkpoint
    info=inspect_checkpoint(path)
    if info["status"]!="ready":raise ValueError(f"checkpoint incomplete: {path}")
    names=["config.json","tokenizer_config.json",*info["weight_files"]]
    info["sha256"]={n:sha256_file(Path(path)/n) for n in names if (Path(path)/n).is_file() and (hash_weights or n not in info["weight_files"])}
    info["weight_hashes_complete"]=hash_weights
    return info


@dataclass(frozen=True)
class ExperimentProtocol:
    split_id: str
    sample_ids: list[str]
    input_condition: str
    checkpoints: dict
    seeds: list[int]
    max_generation_calls: int
    evaluation_version: str
    aggregation: str
    prompt_source: str
    reference_source: str
    renderer: str
    duration_seconds: float
    evaluation_settings: dict = field(default_factory=dict)

    def to_dict(self):
        if not self.split_id or not self.sample_ids or len(set(self.sample_ids))!=len(self.sample_ids):raise ValueError("nonempty unique sample IDs required")
        if self.input_condition not in {"V","V+q","V+r","V+q+r"}:raise ValueError("unknown condition")
        if not self.seeds or len(set(self.seeds))!=len(self.seeds) or self.max_generation_calls<1 or self.duration_seconds<=0:raise ValueError("invalid seeds/budget/duration")
        if not self.evaluation_settings:raise ValueError("record evaluation settings including lag, windows, checkpoints and references")
        return asdict(self)


def write_run_manifest(path,*,protocol,records,source_root,metadata=None):
    config=protocol.to_dict() if isinstance(protocol,ExperimentProtocol) else protocol
    expected={(sid,seed) for sid in config["sample_ids"] for seed in config["seeds"]}
    keys=[(r["sample_id"],r["seed"]) for r in records]
    if len(set(keys))!=len(keys):raise ValueError("duplicate sample/seed records")
    if set(keys)!=expected:raise ValueError("record every planned sample/seed, including explicit failures")
    result={"schema_version":1,"protocol":config,"protocol_id":canonical_hash(config),"records":records,
            "source":source_fingerprint(source_root),"python":platform.python_version(),"metadata":metadata or {}}
    Path(path).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result
