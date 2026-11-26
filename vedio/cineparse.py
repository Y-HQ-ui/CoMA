"""Explicit ViT/GRU reconstruction, not the missing historical CineParse model.

Only local torchvision ViT-B/16 weights are loaded. Five frame predictions are
pooled within each window; measured OpenCV evidence remains a separate stream.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from experiments.provenance import sha256_file


VERSION = "cineparse_vit_gru_reconstruction_v1"
LABELS = ("motion_intensity", "cut_density", "brightness", "color_warmth", "visual_novelty")
PREPROCESS = {"color": "RGB", "resize_short_edge": 256, "center_crop": 224,
              "interpolation": "bilinear", "antialias": True,
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


@dataclass(frozen=True)
class CineParseConfig:
    sample_fps: float = 2.0
    max_frames: int = 32
    adapter_dim: int = 128
    temporal_hidden: int = 64
    trainable_backbone_blocks: int = 0

    def __post_init__(self):
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        for key in ("max_frames", "adapter_dim", "temporal_hidden"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.trainable_backbone_blocks) is not int or not 0 <= self.trainable_backbone_blocks <= 12:
            raise ValueError("trainable_backbone_blocks must be an integer in [0,12]")


def resolve_backbone(path: str = "") -> Path:
    explicit = path or os.environ.get("COMA_VIT_CHECKPOINT", "")
    if explicit:
        result = Path(explicit).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    result = Path(torch.hub.get_dir()) / "checkpoints/vit_b_16-c867db91.pth"
    if not result.is_file():
        raise FileNotFoundError("Local ViT-B/16 checkpoint absent; set COMA_VIT_CHECKPOINT or --backbone-checkpoint")
    return result.resolve()


def load_backbone(path: str, trainable_blocks: int = 0):
    from torchvision.models import vit_b_16
    backbone = vit_b_16(weights=None)  # Never initiates a download.
    backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    backbone.heads = nn.Identity()
    backbone.requires_grad_(False)
    if trainable_blocks:
        for block in list(backbone.encoder.layers.children())[-trainable_blocks:]:
            block.requires_grad_(True)
        backbone.encoder.ln.requires_grad_(True)
    return backbone


class CineParse(nn.Module):
    def __init__(self, backbone: nn.Module, config: CineParseConfig, feature_dim: int = 768):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.adapter = nn.Sequential(nn.LayerNorm(feature_dim * 2 + 1),
                                     nn.Linear(feature_dim * 2 + 1, config.adapter_dim), nn.GELU())
        self.temporal = nn.GRU(config.adapter_dim, config.temporal_hidden,
                               batch_first=True, bidirectional=True)
        self.prior_head = nn.Linear(config.temporal_hidden * 2, len(LABELS))
        self.hit_head = nn.Linear(config.temporal_hidden * 2, 1)

    def train(self, mode=True):
        super().train(mode)
        # Deterministic backbone/dropout behavior, including partially tuned runs.
        self.backbone.eval()
        return self

    def forward(self, frames, lengths, times):
        if frames.ndim != 5 or times.shape != frames.shape[:2]:
            raise ValueError("expected frames [B,T,C,H,W] and times [B,T]")
        b, t = frames.shape[:2]
        lengths = torch.as_tensor(lengths, device=frames.device, dtype=torch.long)
        if lengths.shape != (b,) or (lengths < 1).any() or (lengths > t).any():
            raise ValueError("invalid sequence lengths")
        mask = torch.arange(t, device=frames.device)[None, :] < lengths[:, None]
        valid_times = times[mask]
        if not torch.isfinite(valid_times).all():
            raise ValueError("nonfinite frame timestamps")
        delta_t = torch.zeros_like(times)
        delta_t[:, 1:] = times[:, 1:] - times[:, :-1]
        if (delta_t[:, 1:][mask[:, 1:]] <= 0).any():
            raise ValueError("frame times must increase strictly")
        context = nullcontext() if any(p.requires_grad for p in self.backbone.parameters()) else torch.no_grad()
        with context:
            encoded = self.backbone(frames[mask])
        features = encoded.new_zeros((b, t, self.feature_dim))
        features[mask] = encoded
        difference = torch.zeros_like(features)
        difference[:, 1:] = features[:, 1:] - features[:, :-1]
        # Zero padded inputs before recurrent packing; no padding is encoded/pooled.
        inputs = torch.cat((features, difference, delta_t.unsqueeze(-1)), dim=-1)
        inputs = torch.where(mask.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        packed = pack_padded_sequence(self.adapter(inputs), lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.temporal(packed)
        temporal, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=t)
        frame_priors = self.prior_head(temporal).sigmoid() * mask.unsqueeze(-1)
        pooled = frame_priors.sum(dim=1) / lengths[:, None]
        return {"prior_5d": pooled, "frame_priors": frame_priors,
                "hit_logits": self.hit_head(temporal).squeeze(-1), "mask": mask}


def build_model(config: CineParseConfig, backbone_path: str = "", device="cpu"):
    path = resolve_backbone(backbone_path)
    model = CineParse(load_backbone(str(path), config.trainable_backbone_blocks), config).to(device)
    return model, {"architecture": "torchvision.vit_b_16", "path": str(path),
                   "sha256": sha256_file(path), "preprocess": PREPROCESS}


def read_window(video_path, start_time: float, end_time: float, config: CineParseConfig):
    """Sample the global requested Hz grid, retaining actual decoded frame times."""
    from torchvision.models import ViT_B_16_Weights
    if not all(math.isfinite(x) for x in (start_time, end_time)) or not 0 <= start_time < end_time:
        raise ValueError("invalid window boundaries")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("video must have valid fps and frame count")
        duration = count / fps
        if end_time > duration + 1e-6:
            raise ValueError(f"window ends at {end_time}, beyond video duration {duration}")
        first = math.ceil(start_time * config.sample_fps - 1e-8)
        last = math.ceil(end_time * config.sample_fps - 1e-8)
        if last - first > config.max_frames:
            raise ValueError("window exceeds max_frames; split windows explicitly")
        requested = np.arange(first, last) / config.sample_fps
        indices = np.unique(np.ceil(requested * fps - 1e-8).astype(np.int64))
        indices = indices[(indices < count) & (indices / fps >= start_time - 1e-8)
                          & (indices / fps < end_time)]
        if not len(indices):
            raise ValueError("window contains no frames on the requested sampling grid")
        transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        wanted = set(indices.tolist())
        frames, times = [], []
        for index in range(int(indices[0]), int(indices[-1]) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"video decode failed at frame {index}")
            if index in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(transform(torch.from_numpy(rgb).permute(2, 0, 1)))
                times.append(index / fps)
        return torch.stack(frames), torch.tensor(times, dtype=torch.float64)
    finally:
        cap.release()


def collate_windows(items, device="cpu"):
    lengths = [len(item[0]) for item in items]
    frames = torch.zeros(len(items), max(lengths), *items[0][0].shape[1:])
    times = torch.zeros(len(items), max(lengths), dtype=torch.float64)
    for i, (video, timestamps) in enumerate(items):
        frames[i, :len(video)] = video
        times[i, :len(video)] = timestamps
    # Relative times avoid float32 loss of precision for long videos.
    times = times - times[:, :1]
    return frames.to(device), lengths, times.to(device, dtype=torch.float32)


def save_checkpoint(path, model, base, *, steps, rhythm_steps=0, metadata=None):
    if steps < 1:
        raise ValueError("refusing to export an untrained prediction head")
    state = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    if not all(torch.isfinite(p).all() for p in state.values()):
        raise ValueError("nonfinite trained parameters")
    payload = {"version": VERSION, "config": asdict(model.config), "labels": list(LABELS),
               "base": base, "steps": int(steps), "rhythm_steps": int(rhythm_steps),
               "trainable_state": state, "metadata": metadata or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(checkpoint, *, backbone_path="", device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("version") != VERSION or payload.get("labels") != list(LABELS):
        raise ValueError("incompatible CineParse checkpoint version/label order")
    if payload.get("steps", 0) < 1 or payload["base"].get("preprocess") != PREPROCESS:
        raise ValueError("untrained head or incompatible preprocessing")
    path = resolve_backbone(backbone_path or payload["base"]["path"])
    if sha256_file(path) != payload["base"]["sha256"]:
        raise ValueError("CineParse base weight SHA-256 mismatch")
    config = CineParseConfig(**payload["config"])
    model = CineParse(load_backbone(str(path), config.trainable_backbone_blocks), config)
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    state = payload["trainable_state"]
    if set(state) != set(expected):
        raise ValueError("CineParse trainable parameter names mismatch")
    with torch.no_grad():
        for name, parameter in expected.items():
            value = state[name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid CineParse parameter: {name}")
            parameter.copy_(value)
    model.to(device).eval()
    metadata = {key: value for key, value in payload.items() if key != "trainable_state"}
    metadata.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256_file(checkpoint),
                    resolved_backbone=str(path))
    return model, metadata


def apply_cineparse(pack, checkpoint, *, backbone_path="", device="cpu"):
    """Replace planning priors while keeping measured evaluation evidence intact."""
    import copy
    model, metadata = load_checkpoint(checkpoint, backbone_path=backbone_path, device=device)
    if not pack.get("prior_5d"):
        raise ValueError("no visual windows to predict")
    output = copy.deepcopy(pack)
    output["measured_prior_5d"] = output["prior_5d"]
    output["measured_summary"] = copy.deepcopy(output["summary"])
    output["measurement_extractor"] = output["extractor"]
    output["prior_5d"] = []
    audit = []
    with torch.inference_mode():
        for window in pack["prior_5d"]:
            start, end = window["start_time"], window["end_time"]
            item = read_window(pack["video_path"], start, end, model.config)
            prediction = model(*collate_windows([item], device=device))
            values = prediction["prior_5d"][0].cpu().tolist()
            output["prior_5d"].append({"start_time": start, "end_time": end, **dict(zip(LABELS, values))})
            audit.append({"start_time": start, "end_time": end, "times": item[1].tolist(),
                          "frame_priors": prediction["frame_priors"][0].cpu().tolist(),
                          "hit_probability": prediction["hit_logits"][0].sigmoid().cpu().tolist()
                          if metadata.get("rhythm_steps", 0) else None})
    output["summary"]["means"] = {key: float(np.mean([p[key] for p in output["prior_5d"]])) for key in LABELS}
    output["summary"]["means_scope"] = "unweighted_mean_of_learned_windows"
    output["summary"]["learned_sample_fps"] = model.config.sample_fps
    output["learned_window_predictions"] = audit
    output["extractor"] = {"backend": VERSION, "trainable": True, "mode": "frozen_inference",
                           "checkpoint": metadata, "measured_frame_features_preserved": True,
                           "limitation": "new window-supervised reconstruction, not recovered paper weights; frame predictions are not independently annotated measurements"}
    return output
