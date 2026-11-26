"""Real local ViT smoke checks; synthetic fixtures are NOT research annotations."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.provenance import sha256_file, source_fingerprint
from vedio.cineparse import (LABELS, CineParseConfig, build_model, collate_windows,
                            load_backbone, load_checkpoint, read_window, save_checkpoint)
from vedio.prior_feature_pack import extract_prior_5d
from vedio.train_cineparse import train, supervised_loss


def parameter_hashes(model):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def create_fixtures(root):
    root.mkdir(parents=True, exist_ok=True)
    for split, offset in (("train", 0), ("validation", 40)):
        video = root / f"{split}.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8., (96, 64))
        if not writer.isOpened():
            raise RuntimeError("MJPG writer unavailable")
        for index in range(24):
            frame = np.full((64, 96, 3), 20 + offset + index * 2, dtype=np.uint8)
            cv2.rectangle(frame, (index * 3, 15), (index * 3 + 20, 45),
                          (40 + offset, 100, 240 - offset), -1)
            writer.write(frame)
        writer.release()
        rows = []
        for index in range(2):
            rows.append({"sample_id": f"synthetic_{split}_{index}", "split": split,
                         "dataset": "synthetic_software_fixture", "movie_id": f"synthetic_{split}_video",
                         "video": video.name, "start_time": float(index), "end_time": float(index + 2),
                         "prior_5d": [.35 + .1 * index, .1, .4, .6, .25],
                         "hit_times": [index + .5], "label_source": "invented_software_test_targets",
                         "label_protocol": "synthetic_targets_for_gradient_checks_not_measured_annotations"})
        (root / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def verify(output_dir, device, backbone_checkpoint=""):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory")
    fixtures = output / "fixtures"
    create_fixtures(fixtures)
    training_args = argparse.Namespace(train_manifest=str(fixtures / "train.jsonl"),
        validation_manifest=str(fixtures / "validation.jsonl"), output_dir=str(output / "training"),
        backbone_checkpoint=backbone_checkpoint, device=device, trainable_backbone_blocks=0,
        sample_fps=2., max_frames=32, epochs=1, steps=2, batch_size=1,
        learning_rate=1e-4, hit_weight=.2, seed=123)
    completion = train(training_args)
    checkpoint = output / "training/cineparse.pt"
    model, metadata = load_checkpoint(checkpoint, device=device)
    # torchvision migrates old MLP parameter names while loading this checkpoint.
    original = load_backbone(metadata["resolved_backbone"]).state_dict()
    frozen_unchanged = all(torch.equal(p.detach().cpu(), original[name])
                           for name, p in model.backbone.named_parameters())
    assert frozen_unchanged
    del original
    item = read_window(fixtures / "train.avi", 0., 2., model.config)
    with torch.inference_mode():
        expected = model(*collate_windows([item], device))["prior_5d"].cpu()
    save_checkpoint(output / "roundtrip.pt", model, metadata["base"], steps=metadata["steps"],
                    rhythm_steps=metadata["rhythm_steps"], metadata=metadata["metadata"])
    del model
    gc.collect()
    restored, _ = load_checkpoint(output / "roundtrip.pt", device=device)
    with torch.inference_mode():
        actual = restored(*collate_windows([item], device))["prior_5d"].cpu()
    assert torch.equal(expected, actual)
    del restored
    gc.collect()

    video = str(fixtures / "train.avi")
    baseline = extract_prior_5d(video)
    learned = extract_prior_5d(video, cineparse_checkpoint=str(checkpoint), cineparse_device=device)
    assert baseline["frame_features"] == learned["frame_features"]
    assert baseline["prior_5d"] == learned["measured_prior_5d"]
    assert [(p["start_time"], p["end_time"]) for p in baseline["prior_5d"]] == \
           [(p["start_time"], p["end_time"]) for p in learned["prior_5d"]]
    assert all(0 <= p[k] <= 1 for p in learned["prior_5d"] for k in LABELS)
    (output / "prior_opencv.json").write_text(json.dumps(baseline, indent=2))
    (output / "prior_learned.json").write_text(json.dumps(learned, indent=2))

    # Independently exercise optional fine-tuning on the actual final ViT block.
    torch.manual_seed(123)
    tuned, base = build_model(CineParseConfig(trainable_backbone_blocks=1), backbone_checkpoint, device)
    tuned.train()
    before = parameter_hashes(tuned)
    inputs = collate_windows([item], device)
    prediction = tuned(*inputs)
    target = torch.tensor([[.35, .1, .4, .6, .25]], device=device)
    hits = torch.tensor([[0., 1., 0., 0.]], device=device)
    optimizer = torch.optim.AdamW([p for p in tuned.parameters() if p.requires_grad], lr=1e-4)
    loss, _, _ = supervised_loss(prediction, target, hits, prediction["mask"])
    loss.backward()
    assert all(p.grad is None for p in tuned.parameters() if not p.requires_grad)
    grad = float(torch.nn.utils.clip_grad_norm_([p for p in tuned.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
    optimizer.step()
    after = parameter_hashes(tuned)
    changed = [name for name in before if before[name] != after[name]]
    trainables = {name for name, p in tuned.named_parameters() if p.requires_grad}
    assert changed and set(changed) <= trainables
    backbone_changes = [name for name in changed if name.startswith("backbone.")]
    assert backbone_changes and all(name.startswith(("backbone.encoder.layers.encoder_layer_11.",
                                                    "backbone.encoder.ln.")) for name in backbone_changes)
    tuned.eval()
    with torch.inference_mode():
        tuned_expected = tuned(*inputs)["prior_5d"].cpu()
    save_checkpoint(output / "partial_finetune.pt", tuned, base, steps=1, rhythm_steps=1,
                    metadata={"purpose": "one synthetic step, not a trained research model"})
    del tuned, optimizer, prediction, loss
    gc.collect()
    reloaded, _ = load_checkpoint(output / "partial_finetune.pt", device=device)
    with torch.inference_mode():
        tuned_actual = reloaded(*inputs)["prior_5d"].cpu()
    assert torch.equal(tuned_expected, tuned_actual)
    report = {"status": "passed", "purpose": "synthetic software verification, not paper accuracy",
              "device": device, "base": base, "training": completion,
              "frozen_backbone_equals_original": frozen_unchanged,
              "checkpoint_reload_bit_identical": True, "measurement_features_preserved": True,
              "window_boundaries_preserved": True, "window_count": len(learned["prior_5d"]),
              "actual_sample_times": item[1].tolist(),
              "partial_finetuning": {"changed_tensors": changed,
                  "changed_backbone_tensor_count": len(backbone_changes), "frozen_tensors_changed": 0,
                  "grad_norm": grad, "checkpoint_reload_bit_identical": True},
              "checkpoint_sha256": sha256_file(checkpoint),
              "source": source_fingerprint(Path(__file__).resolve().parents[1])}
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key not in {"source", "partial_finetuning"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-checkpoint", default="")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir, args.device, args.backbone_checkpoint), indent=2))


if __name__ == "__main__":
    main()
