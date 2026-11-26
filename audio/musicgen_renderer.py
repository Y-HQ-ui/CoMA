from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class MusicGenRendererSession:
    _instances: Dict[tuple[str, str], "MusicGenRendererSession"] = {}

    def __init__(self, *, model_dir: str, device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from model_registry import resolve_model
        model_dir = resolve_model("musicgen", model_dir)
        self.model_dir = model_dir
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_dir, local_files_only=True, torch_dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.sample_rate = int(self.model.config.audio_encoder.sampling_rate)

    @classmethod
    def get(cls, *, model_dir: str, device: Optional[str] = None) -> "MusicGenRendererSession":
        resolved_device = device or "auto"
        key = (model_dir, resolved_device)
        if key not in cls._instances:
            cls._instances[key] = cls(model_dir=model_dir, device=device)
        return cls._instances[key]

    def render_one(
        self,
        prompt: str,
        *,
        output_path: str,
        seed: int = 123,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 250,
        guidance_scale: float = 3.0,
        max_new_tokens: int = 1500,
        duration: Optional[float] = None,
        temporal_features=None,
        adapter_checkpoint: str = "",
    ) -> str:
        import soundfile as sf
        import torch

        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)
        from contextlib import nullcontext
        if duration is not None:
            if not 0 < duration <= 30:
                raise ValueError("MusicGen segment duration must be in (0,30] seconds")
            frame_rate = float(self.model.config.audio_encoder.frame_rate)
            max_new_tokens = int(np.ceil(duration*frame_rate)) + self.model.decoder.config.num_codebooks
        context = nullcontext()
        if adapter_checkpoint:
            if temporal_features is None:
                raise ValueError("temporal adapter requires score/control features")
            from audio.temporal_control import TemporalControlAdapter, musicgen_temporal_context
            adapter = TemporalControlAdapter.load(adapter_checkpoint,self.device,expected_base_model=self.model_dir).eval()
            context = musicgen_temporal_context(self.model,adapter,temporal_features)
        with torch.no_grad(), context:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                guidance_scale=float(guidance_scale),
            )
        audio = audio_values[0].detach().float().cpu().numpy()
        waveform = audio[0] if audio.ndim == 2 and audio.shape[0] == 1 else audio.mean(axis=0)
        if duration is not None:
            target = round(duration*self.sample_rate)
            if waveform.size < target:
                raise RuntimeError(f"MusicGen generated {waveform.size} samples, expected at least {target}")
            waveform = waveform[:target]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), waveform, self.sample_rate)
        del inputs, audio_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return str(out)


def render_musicgen_prompts(
    prompts: List[str],
    *,
    model_dir: str,
    output_dir: str,
    device: Optional[str] = None,
    seed: int = 123,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 250,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 1500,
    concatenate: bool = True,
    controls: Optional[List[Dict[str, object]]] = None,
    symbolic_outputs=None,
    adapter_checkpoint: str = "",
    budget=None,
) -> Dict[str, object]:
    from model_registry import resolve_model
    model_dir = resolve_model("musicgen", model_dir)
    if controls is not None and len(controls) != len(prompts):
        raise ValueError("controls must match prompts")

    import soundfile as sf

    renderer = MusicGenRendererSession.get(model_dir=model_dir, device=device)
    run_device = renderer.device
    out_dir = Path(output_dir)
    segment_dir = out_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = renderer.sample_rate

    segment_paths: List[str] = []
    rendered_audio: List[np.ndarray] = []
    for index, prompt in enumerate(prompts):
        segment_path = segment_dir / f"segment_{index:04d}.wav"
        duration = None
        temporal_features = None
        if controls is not None:
            import copy
            control = copy.deepcopy(controls[index])
            start = float(control["start_time"])
            duration = float(control["end_time"])-start
            control["start_time"], control["end_time"] = 0.0,duration
            for window in control.get("render_windows",[]):
                window["start_time"] -= start; window["end_time"] -= start
            events=[]
            for item in symbolic_outputs or []:
                for event in item.get("events",[]):
                    end = event["onset"]+event["duration"]
                    if end > start and event["onset"] < start+duration:
                        event=dict(event); event["onset"]=max(0,event["onset"]-start)
                        event["duration"]=min(duration,end-start)-event["onset"]; events.append(event)
            from audio.temporal_control import build_temporal_features
            temporal_features = build_temporal_features([control],events,duration=duration)
        if budget is not None: budget.consume("musicgen")
        renderer.render_one(
            prompt,
            output_path=str(segment_path),
            seed=seed + index,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens,
            duration=duration,
            temporal_features=temporal_features,
            adapter_checkpoint=adapter_checkpoint,
        )
        waveform, _ = sf.read(str(segment_path), always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        segment_paths.append(str(segment_path))
        rendered_audio.append(np.asarray(waveform, dtype=np.float32))

    soundtrack_path = ""
    if concatenate and rendered_audio:
        if controls is not None:
            from audio.timeline import assemble_timeline
            soundtrack = assemble_timeline(rendered_audio,[(c["start_time"],c["end_time"]) for c in controls],sample_rate)
        else:
            soundtrack = np.concatenate(rendered_audio)
        final_path = out_dir / "soundtrack.wav"
        sf.write(str(final_path), soundtrack, sample_rate)
        soundtrack_path = str(final_path)

    manifest = {
        "model_dir": model_dir,
        "device": run_device,
        "sample_rate": sample_rate,
        "segment_paths": segment_paths,
        "soundtrack_path": soundtrack_path,
        "prompts": prompts,
        "seed": seed,
        "conditioning_backend": "temporal_adapter" if adapter_checkpoint else "text_only",
        "adapter_checkpoint": adapter_checkpoint or None,
        "intervals": [[c["start_time"],c["end_time"]] for c in controls] if controls else None,
        "duration_seconds": len(soundtrack)/sample_rate if concatenate and rendered_audio else None,
        "generation_calls": len(segment_paths),
        "transition": "10ms fades within fixed intervals" if controls else "legacy_concatenation",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = ["MusicGenRendererSession", "render_musicgen_prompts"]
