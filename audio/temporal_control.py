"""Trainable temporal residual at MusicGen's summed decoder embeddings.

Input [B,T,29] at codec frame rate (50 Hz for standard MusicGen); projected
residual [B,T,H] is added BEFORE the first decoder block. Base weights remain
frozen. Output projection starts at zero. Checkpoints are stored separately.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn

ROLES = ("melody", "harmony", "bass", "drums", "pad", "arpeggio", "fx")
FEATURES = ("tempo/300", "energy", "motion", "cuts", "brightness", "warmth", "novelty",
            "score_onset", "score_energy", "mean_pitch/127", *[f"chroma_{i}" for i in range(12)],
            *[f"density_{r}" for r in ROLES])


def build_temporal_features(controls, events=(), *, duration=None, frame_rate=50.0):
    if not controls or frame_rate <= 0:
        raise ValueError("controls and positive frame_rate required")
    duration = duration if duration is not None else max(c["end_time"] for c in controls)
    if duration <= 0:
        raise ValueError("duration must be positive")
    n = int(np.ceil(duration*frame_rate))
    features = np.zeros((n, len(FEATURES)), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    times = np.arange(n)/frame_rate
    for c in controls:
        mask = (times >= c["start_time"]) & (times < c["end_time"])
        p = c.get("video_prior_5d", {})
        vector = [c.get("tempo_bpm",120)/300, c.get("energy",.5),
                  *[p.get(k,.5) for k in ("motion_intensity","cut_density","brightness","color_warmth","visual_novelty")]]
        features[mask,:7] += np.asarray(vector)
        counts[mask] += 1
        for track in c.get("tracks", []):
            features[mask,22+ROLES.index(track["role"])] += track["density"]
        for w in c.get("render_windows", []):
            wm = mask & (times >= w["start_time"]) & (times < w["end_time"])
            features[wm,0] += (w.get("tempo_bpm",c.get("tempo_bpm",120))-c.get("tempo_bpm",120))/300
            features[wm,1] += w.get("energy",c.get("energy",.5))-c.get("energy",.5)
            features[wm,4] += w.get("brightness_hint",p.get("brightness",.5))-p.get("brightness",.5)
    if np.any(counts == 0):
        raise ValueError("temporal controls do not cover requested duration")
    features[:,:7] /= counts[:,None]
    features[:,22:] /= counts[:,None]
    pitch_counts = np.zeros(n)
    for e in events:
        e = e.to_dict() if hasattr(e,"to_dict") else e
        left = max(0,int(np.floor(e["onset"]*frame_rate)))
        right = min(n,max(left+1,int(np.ceil((e["onset"]+e["duration"])*frame_rate))))
        if left >= n: continue
        features[left,7] = 1
        features[left:right,8] += e.get("velocity",.7)
        features[left:right,9] += e["pitch"]/127
        features[left:right,10+e["pitch"]%12] += 1
        pitch_counts[left:right] += 1
    active = pitch_counts > 0
    features[active,8:22] /= pitch_counts[active,None]
    if not np.isfinite(features).all():
        raise ValueError("nonfinite temporal controls")
    return features


@dataclass
class AdapterConfig:
    hidden_size: int
    bottleneck: int = 128
    feature_dim: int = len(FEATURES)
    frame_rate: float = 50.0
    version: str = "coma_temporal_v1"


class TemporalControlAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim,config.bottleneck),
                                     nn.SiLU(), nn.Linear(config.bottleneck,config.hidden_size))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features.to(next(self.parameters()).dtype))

    def save(self, path, *, metadata):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config":asdict(self.config),"state_dict":self.state_dict(),"metadata":metadata},path)

    @classmethod
    def load(cls, path, device="cpu", *, expected_base_model=""):
        state = torch.load(path, map_location=device, weights_only=True)
        recorded=state.get("metadata",{}).get("base_model",{}).get("path")
        if recorded and expected_base_model:
            from pathlib import Path
            if Path(recorded).resolve()!=Path(expected_base_model).resolve():
                raise ValueError("temporal adapter checkpoint belongs to a different MusicGen base")
        model = cls(AdapterConfig(**state["config"])).to(device)
        model.load_state_dict(state["state_dict"])
        return model


@contextmanager
def musicgen_temporal_context(model, adapter, features, *, start_frame=0):
    """Works with cached generation, full teacher forcing and CFG batch doubling.

    Decoder step zero is BOS; subsequent steps condition frame step-1. MusicGen
    delays codebooks, so this is a shared decoder-clock residual, not a claim of
    independently aligned conditioning for each delayed codebook.
    """
    decoder = model.decoder.model.decoder
    if adapter.config.hidden_size != decoder.config.hidden_size:
        raise ValueError("adapter hidden size does not match MusicGen")
    if abs(adapter.config.frame_rate-model.config.audio_encoder.frame_rate)>1e-6:
        raise ValueError("adapter and MusicGen codec frame rates differ")
    f = torch.as_tensor(features,device=next(adapter.parameters()).device,dtype=torch.float32)
    if f.ndim == 2: f = f.unsqueeze(0)
    if f.ndim != 3 or f.shape[-1] != adapter.config.feature_dim or f.shape[1] == 0:
        raise ValueError("features must be [B,T,F]")
    def inject(module, args, kwargs, output):
        # Preserve MusicGen's native ID path, including positional embeddings,
        # attention masks and cache offsets. In transformers 4.36 substituting
        # inputs_embeds changes the shape seen by embed_positions (T becomes 1).
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is None or ids.ndim != 3 or ids.shape[1] != decoder.num_codebooks:
            raise ValueError("temporal conditioning requires native decoder token IDs")
        offset = args[1] if len(args) > 1 else kwargs.get("past_key_values_length", 0)
        positions = (torch.arange(ids.shape[-1],device=f.device)+offset-1+start_frame).clamp(0,f.shape[1]-1)
        residual = adapter(f[:,positions])
        if ids.shape[0] % residual.shape[0]:
            raise ValueError("control batch does not match decoder batch")
        residual = residual.repeat(ids.shape[0]//residual.shape[0],1,1)
        return output + residual.to(output)
    handle = decoder.embed_positions.register_forward_hook(inject, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
