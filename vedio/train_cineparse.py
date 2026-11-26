"""Supervised training for the explicitly reconstructed CineParse backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from experiments.provenance import sha256_file, source_fingerprint
from vedio.cineparse import (VERSION, LABELS, CineParseConfig, build_model,
                            collate_windows, read_window, save_checkpoint)


@dataclass
class VisualSample:
    sample_id: str
    split: str
    dataset: str
    movie_id: str
    video: str
    video_sha256: str
    start_time: float
    end_time: float
    prior_5d: list[float]
    label_source: str
    label_protocol: str
    hit_times: list[float] | None = None
    music_work_id: str = ""


def load_manifests(train_manifest, validation_manifest=None):
    """Hash videos independently and reject movie/file/work cross-split leakage."""
    result = {"train": [], "validation": []}
    ids, owners, hashes, windows, protocols = set(), {}, {}, set(), set()
    for split, filename in (("train", train_manifest), ("validation", validation_manifest)):
        if not filename:
            continue
        path = Path(filename).resolve()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prefix = f"{path.name}:{line_number}"
            for key in ("sample_id", "dataset", "movie_id", "video", "label_source", "label_protocol"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{prefix}: nonempty {key} required")
            if row.get("split") != split:
                raise ValueError(f"{prefix}: requires split={split}; test data cannot train/select models")
            if row["sample_id"] in ids:
                raise ValueError(f"{prefix}: duplicate sample_id")
            ids.add(row["sample_id"])
            video = Path(row["video"])
            video = (video if video.is_absolute() else path.parent / video).resolve()
            if not video.is_file():
                raise FileNotFoundError(video)
            if str(video) not in hashes:
                hashes[str(video)] = sha256_file(video)
            digest = hashes[str(video)]
            if row.get("video_sha256") and row["video_sha256"] != digest:
                raise ValueError(f"{prefix}: declared video SHA-256 mismatch")
            for key, value in (("movie_id", row["movie_id"]), ("video_sha256", digest),
                               ("music_work_id", row.get("music_work_id", ""))):
                if not value:
                    continue
                identity = (key, value)
                if identity in owners and owners[identity] != split:
                    raise ValueError(f"{prefix}: {key} crosses train/validation")
                owners[identity] = split
            start, end = float(row["start_time"]), float(row["end_time"])
            if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
                raise ValueError(f"{prefix}: invalid window times")
            identity = (digest, start, end)
            if identity in windows:
                raise ValueError(f"{prefix}: duplicate video window")
            windows.add(identity)
            protocols.add(row["label_protocol"])
            if len(protocols) > 1:
                raise ValueError("all labels must use the same declared normalization/annotation protocol")
            target = row["prior_5d"]
            if isinstance(target, dict):
                if set(target) != set(LABELS):
                    raise ValueError(f"{prefix}: expected exactly five named priors")
                target = [target[key] for key in LABELS]
            target = np.asarray(target, dtype=np.float64)
            if target.shape != (5,) or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
                raise ValueError(f"{prefix}: five finite labels in [0,1] required")
            hits = row.get("hit_times")
            if hits is not None:
                if not isinstance(hits, list):
                    raise ValueError(f"{prefix}: hit_times must be a list in absolute video seconds")
                hits = [float(x) for x in hits]
                if any(not math.isfinite(x) or not start <= x < end for x in hits):
                    raise ValueError(f"{prefix}: hit outside annotated window")
                if hits != sorted(set(hits)):
                    raise ValueError(f"{prefix}: hit_times must be sorted and unique")
            result[split].append(VisualSample(row["sample_id"], split, row["dataset"], row["movie_id"],
                str(video), digest, start, end, target.tolist(), row["label_source"], row["label_protocol"],
                hits, row.get("music_work_id", "")))
        if not result[split]:
            raise ValueError(f"empty {split} manifest")
    return result


def prepare_batch(samples, config, device):
    items = [read_window(s.video, s.start_time, s.end_time, config) for s in samples]
    model_inputs = collate_windows(items, device=device)
    labels = torch.tensor([s.prior_5d for s in samples], dtype=torch.float32, device=device)
    hit_targets = torch.zeros(len(samples), max(len(item[0]) for item in items), device=device)
    hit_mask = torch.zeros_like(hit_targets, dtype=torch.bool)
    for i, (sample, (_, times)) in enumerate(zip(samples, items)):
        if sample.hit_times is not None:
            hit_mask[i, :len(times)] = True
            for hit in sample.hit_times:
                hit_targets[i, int(torch.argmin(torch.abs(times - hit)))] = 1.0
    return model_inputs, labels, hit_targets, hit_mask, [item[1].tolist() for item in items]


def supervised_loss(output, labels, hit_targets, hit_mask, hit_weight=0.2):
    prior = F.mse_loss(output["prior_5d"], labels)
    hit_mask = hit_mask & output["mask"]
    hit = F.binary_cross_entropy_with_logits(output["hit_logits"][hit_mask], hit_targets[hit_mask]) \
        if hit_mask.any() else prior.new_zeros(())
    return prior + hit_weight * hit, prior, hit


def evaluate(model, samples, device, batch_size):
    model.eval()
    records, absolute, squared, movie_errors = [], [], [], {}
    with torch.inference_mode():
        for offset in range(0, len(samples), batch_size):
            batch = samples[offset:offset + batch_size]
            inputs, labels, hits, mask, times = prepare_batch(batch, model.config, device)
            output = model(*inputs)
            for i, sample in enumerate(batch):
                predicted = output["prior_5d"][i].cpu().numpy()
                error = predicted - np.array(sample.prior_5d)
                absolute.append(np.abs(error))
                squared.append(error ** 2)
                mse = float(np.mean(error ** 2))
                movie_errors.setdefault(sample.movie_id, []).append(mse)
                records.append({"sample_id": sample.sample_id, "movie_id": sample.movie_id,
                                "times": times[i], "target": sample.prior_5d,
                                "prediction": predicted.tolist(), "mse": mse})
    return {"window_count": len(samples), "movie_count": len(movie_errors),
            "window_mse": float(np.mean(squared)),
            "movie_mean_mse": float(np.mean([np.mean(v) for v in movie_errors.values()])),
            "mae_by_prior": dict(zip(LABELS, np.mean(absolute, axis=0).tolist())),
            "records": records}


def train(args):
    if args.epochs < 1 or args.steps < 1 or args.batch_size < 1:
        raise ValueError("epochs, steps and batch_size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(args.hit_weight) or args.hit_weight < 0:
        raise ValueError("hit_weight must be finite and nonnegative")
    config = CineParseConfig(sample_fps=args.sample_fps, max_frames=args.max_frames,
                            trainable_backbone_blocks=args.trainable_backbone_blocks)
    samples = load_manifests(args.train_manifest, args.validation_manifest)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    model, base = build_model(config, args.backbone_checkpoint, args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"version": VERSION, "scope": "new supervised reconstruction, not historical paper model",
                "config": asdict(config), "arguments": vars(args), "base": base,
                "source": source_fingerprint(Path(__file__).resolve().parents[1]),
                "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
                "trainable_count": sum(p.numel() for p in parameters),
                "total_count": sum(p.numel() for p in model.parameters()),
                "optimizer": {"name": "AdamW", "weight_decay": 0.01, "grad_clip": 1.0},
                "loss": "window-prior MSE + hit_weight * optional frame BCE (only annotated windows)",
                "hit_quantization": "nearest sampled frame within window, absolute seconds; collisions merge; no finer-time claim",
                "selection": "validation movie_mean_mse" if samples["validation"] else "last_step_no_validation",
                "checkpoint_scope": "trainable parameters only; no optimizer/RNG resume",
                "data": {split: [asdict(s) for s in rows] for split, rows in samples.items()},
                "manifest_sha256": {"train": sha256_file(args.train_manifest),
                    "validation": sha256_file(args.validation_manifest) if args.validation_manifest else None},
                "leakage_audit": "movie IDs, file hashes and supplied work IDs; perceptual duplicates/pretraining overlap not audited"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    step, rhythm_steps, best = 0, 0, math.inf
    with (output / "training.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            model.train()
            order = list(samples["train"])
            rng.shuffle(order)
            for offset in range(0, len(order), args.batch_size):
                batch = order[offset:offset + args.batch_size]
                inputs, labels, hits, hit_mask, times = prepare_batch(batch, config, args.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(*inputs)
                loss, prior_loss, hit_loss = supervised_loss(prediction, labels, hits, hit_mask, args.hit_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite CineParse loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                rhythm_steps += int(hit_mask.any() and args.hit_weight > 0)
                record = {"step": step, "epoch": epoch, "sample_ids": [s.sample_id for s in batch],
                          "times": times, "loss": float(loss.detach()), "prior_mse": float(prior_loss.detach()),
                          "hit_bce": float(hit_loss.detach()), "grad_norm": float(grad_norm)}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                if step >= args.steps:
                    break
            metadata = {"protocol": "protocol.json", "seed": args.seed, "epoch": epoch,
                        "selection": protocol["selection"],
                        "label_sources": sorted({s.label_source for s in samples["train"]}),
                        "label_protocols": sorted({s.label_protocol for s in samples["train"]})}
            if samples["validation"]:
                validation = evaluate(model, samples["validation"], args.device, args.batch_size)
                (output / f"validation_epoch_{epoch:04d}.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
                metadata["validation_movie_mean_mse"] = validation["movie_mean_mse"]
                improved = validation["movie_mean_mse"] < best
                best = min(best, validation["movie_mean_mse"])
            else:
                improved = True
            save_checkpoint(output / "last.pt", model, base, steps=step, rhythm_steps=rhythm_steps, metadata=metadata)
            if improved:
                save_checkpoint(output / "cineparse.pt", model, base, steps=step, rhythm_steps=rhythm_steps, metadata=metadata)
            if step >= args.steps:
                break
    completion = {"status": "completed", "steps": step, "rhythm_steps": rhythm_steps,
                  "stop": "step_budget" if step >= args.steps else "epoch_budget",
                  "checkpoint": str(output / "cineparse.pt"), "selection": protocol["selection"],
                  "best_validation_movie_mean_mse": best if math.isfinite(best) else None}
    (output / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone-checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trainable-backbone-blocks", type=int, default=0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hit-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
