from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planning import load_blueprint_json
from reward.config import build_default_rl_config
from reward.evidence import RewardEvidenceEvaluator
from reward.hierarchical_refine import refine_blueprint_hierarchical


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def find_blueprints(inputs: Iterable[str], max_clips: int) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("blueprint.json")))
            paths.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            paths.append(path)
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique[:max_clips]


def flatten_weight_history(result: Dict[str, Any]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for item in result.get("training_history", {}).get("weight_history", []):
        weights = item.get("weights", {})
        alpha = np.asarray(weights.get("alpha_vis", []), dtype=np.float64)
        beta = np.asarray(weights.get("beta_mus", []), dtype=np.float64)
        row = {"step": float(item.get("step", len(rows)))}
        row.update({f"alpha_vis_{index}": float(value) for index, value in enumerate(alpha)})
        row.update({f"beta_mus_{index}": float(value) for index, value in enumerate(beta)})
        row["gamma_cons"] = float(weights.get("gamma_cons", 0.0))
        row["visual_group"] = float(alpha.sum())
        row["musicality_group"] = float(beta.sum())
        rows.append(row)
    return rows


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if key not in {"step", "run"}})
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.size),
        }
    return summary


def run_suite(
    blueprint_paths: List[Path],
    *,
    output_dir: Path,
    runs: int,
    episodes: int,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    if not blueprint_paths:
        raise ValueError("no blueprint JSON files found")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    all_weight_rows: List[Dict[str, float]] = []
    for run_index in range(runs):
        run_seed = seed + run_index
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        blueprint_path = blueprint_paths[run_index % len(blueprint_paths)]
        blueprint = load_blueprint_json(str(blueprint_path))
        config = build_default_rl_config()
        config.num_episodes = episodes
        result = refine_blueprint_hierarchical(
            blueprint,
            config=config,
            num_episodes=episodes,
            device=device,
            reward_evaluator=RewardEvidenceEvaluator(
                temporal_alignment_freq=config.temporal_alignment_freq,
            ),
        )
        run_path = output_dir / f"run_{run_index:03d}.json"
        run_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        weight_rows = flatten_weight_history(result)
        for row in weight_rows:
            row["run"] = float(run_index)
        all_weight_rows.extend(weight_rows)
        reward_history = result.get("training_history", {}).get("reward_components", [])
        run_records.append(
            {
                "run": run_index,
                "seed": run_seed,
                "blueprint": str(blueprint_path),
                "result": str(run_path),
                "final_reward": reward_history[-1] if reward_history else {},
            }
        )
    summary = {
        "requested_runs": runs,
        "completed_runs": len(run_records),
        "clip_count": len(blueprint_paths),
        "episodes_per_run": episodes,
        "runs": run_records,
        "weight_statistics": summarize_rows(all_weight_rows),
        "weight_trajectories": all_weight_rows,
    }
    summary_path = output_dir / "supplement_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_supplement_training")
    parser.add_argument("inputs", nargs="+", help="Blueprint JSON files or directories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    paths = find_blueprints(args.inputs, max(1, args.max_clips))
    summary = run_suite(
        paths,
        output_dir=Path(args.output_dir),
        runs=max(1, args.runs),
        episodes=max(1, args.episodes),
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({"completed_runs": summary["completed_runs"], "clip_count": summary["clip_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
