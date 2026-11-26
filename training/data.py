"""Manifest-based supervised data; explicit train split and immutable sample IDs."""
import json
from pathlib import Path
import torch


def read_training_manifest(path,stage):
    path=Path(path);records=[];seen=set()
    for line_number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        if row.get("split")!="train":raise ValueError(f"line {line_number}: training can only consume split=train")
        if not row.get("sample_id") or row["sample_id"] in seen:raise ValueError("sample_id missing/duplicated")
        seen.add(row["sample_id"])
        required=(("blueprint", "audio", "symbolic_outputs") if stage == "audio-sft" else
                  ("blueprint", "symbolic_outputs") if stage == "audio-ppo" else
                  ("blueprint",) if stage == "guidance" else () if stage == "symbolic-ppo" else ("abc",))
        required=tuple(dict.fromkeys((*required, *[key for key in ("blueprint", "reward_inputs", "audio", "symbolic_outputs") if row.get(key)])))
        for key in required:
            p=Path(row[key]);p=p if p.is_absolute() else path.parent/p
            if not p.is_file():raise FileNotFoundError(p)
            row[key]=str(p.resolve())
        records.append(row)
    if not records:raise ValueError("empty training manifest")
    return records


def symbolic_example(row,tokenizer,device,max_tokens=4096,*,supervised=True,max_continuation_tokens=2048):
    if not supervised:
        prefix = row.get("prefix", "").replace("\n", "<n>")
        if not prefix.strip():
            raise ValueError("symbolic PPO requires an explicit prefix")
        prompt = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if not prompt or len(prompt) >= max_tokens:
            raise ValueError("empty/oversize symbolic PPO condition")
        return {}, torch.tensor([prompt], device=device)
    text=Path(row["abc"]).read_text().replace("<n>","\n")
    lines=text.splitlines(keepends=True)
    key_line=next((i for i,l in enumerate(lines) if l.startswith("K:")),None)
    if key_line is None:raise ValueError("supervised ABC requires key header")
    prefix="".join(lines[:key_line+1]).replace("\n","<n>")
    prefix=str(row.get("prefix",prefix)).replace("\n","<n>")
    target="".join(lines[key_line+1:]).replace("\n","<n>")
    prompt=tokenizer(prefix,add_special_tokens=False)["input_ids"]
    continuation=tokenizer(target,add_special_tokens=False)["input_ids"]+[tokenizer.eos_token_id]
    if not prompt or not continuation or len(prompt)+len(continuation)>max_tokens or len(continuation)>max_continuation_tokens:
        raise ValueError("empty/oversize sample; segment the dataset explicitly, no silent truncation")
    tokens=torch.tensor([prompt+continuation],device=device)
    mask=torch.zeros_like(tokens[:,:-1],dtype=torch.bool);mask[:,len(prompt)-1:]=True
    return {"tokens":tokens,"action_mask":mask},torch.tensor([prompt],device=device)


def audio_example(row,model,processor,device,*,supervised=True):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    from planning import load_blueprint_json
    from planning.timeline import merge_overlapping_segments
    from audio.abc_to_render_controls import blueprint_to_render_controls
    from audio.musicgen_adapter import render_controls_to_prompts
    from audio.temporal_control import build_temporal_features
    from .policies import delayed_audio_tokens
    blueprint=merge_overlapping_segments(load_blueprint_json(row["blueprint"]))
    symbolic=json.loads(Path(row["symbolic_outputs"]).read_text()) if row.get("symbolic_outputs") else []
    if not symbolic:raise ValueError("audio warm start requires aligned symbolic_outputs with note events")
    controls=blueprint_to_render_controls(blueprint,symbolic_outputs=symbolic)
    duration=max(s.end_time for s in blueprint.segments)
    if not 0 < duration <= 30:raise ValueError("training clips must be in (0,30] seconds")
    events=[e for item in symbolic for e in item["events"]]
    features=torch.tensor(build_temporal_features(controls,events,duration=duration),device=device)[None]
    inputs=processor(text=[row.get("prompt") or "; ".join(render_controls_to_prompts(controls))],return_tensors="pt",padding=True).to(device)
    payload={"features":features,"text_inputs":dict(inputs)}
    # On-policy audio training consumes generated conditions only. Do not open or
    # encode a target waveform merely to obtain dimensions or an unused label.
    if not supervised:
        return payload,blueprint,symbolic
    waveform,sr=sf.read(row["audio"],always_2d=True);waveform=waveform.mean(1)
    target_sr=model.config.audio_encoder.sampling_rate
    if sr!=target_sr:
        factor=gcd(sr,target_sr);waveform=resample_poly(waveform,target_sr//factor,sr//factor)
    if abs(len(waveform)/target_sr-duration)>.05:raise ValueError("audio/blueprint duration mismatch")
    audio=torch.tensor(waveform,dtype=next(model.parameters()).dtype,device=device)[None,None]
    with torch.no_grad(): codes=model.audio_encoder.encode(audio).audio_codes
    if codes.shape[0]!=1:raise ValueError("chunked codec output unsupported")
    if codes.shape[2] != model.decoder.num_codebooks:codes=codes[:,:,:model.decoder.num_codebooks]
    tokens,mask=delayed_audio_tokens(codes[0],model.decoder.config.pad_token_id,model.decoder.config.audio_channels)
    return {**payload,"tokens":tokens,"codebook_mask":mask},blueprint,symbolic
