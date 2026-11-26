"""Run supervised warm start or generator PPO. All checkpoints go to a new run directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch

from .token_ppo import PPOConfig,TokenPPOTrainer,temporal_credit
from .policies import SymbolicTokenPolicy,AudioTokenPolicy
from .data import read_training_manifest,symbolic_example,audio_example


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=["symbolic-sft","audio-sft","symbolic-ppo","audio-ppo"])
    p.add_argument("--manifest",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--model-dir",default="");p.add_argument("--warm-start-checkpoint",default="")
    p.add_argument("--resume-checkpoint",default="")
    p.add_argument("--adapter-checkpoint",default="");p.add_argument("--device",default="cpu")
    p.add_argument("--epochs",type=int,default=1);p.add_argument("--steps",type=int,default=100)
    p.add_argument("--seed",type=int,default=123);p.add_argument("--max-new-tokens",type=int,default=256)
    p.add_argument("--learning-rate",type=float,default=1e-5);p.add_argument("--disable-consistency",action="store_true")
    p.add_argument("--reward-backend",choices=["paper","artifact_proxy"],default="paper")
    p.add_argument("--paper-config",default="")
    p.add_argument("--ppo-epochs",type=int,default=4)
    p.add_argument("--ppo-clip",type=float,default=.2)
    p.add_argument("--value-clip",type=float,default=.2)
    p.add_argument("--gae-lambda",type=float,default=.95,help="GAE decay per declared discount unit")
    p.add_argument("--entropy-coef",type=float,default=.01)
    p.add_argument("--value-coef",type=float,default=.5)
    p.add_argument("--max-grad-norm",type=float,default=1.)
    p.add_argument("--target-kl",type=float,default=.03)
    p.add_argument("--discount-per-unit",type=float,default=None,help="default .95 per symbolic phrase or .98 per audio window")
    p.add_argument("--discount-unit-seconds",type=float,default=None)
    p.add_argument("--audio-window-seconds",type=float,default=.4)
    p.add_argument("--skip-weight-hash",action="store_true")
    p.add_argument("--profile-memory",action="store_true")
    args=p.parse_args(argv)
    if args.profile_memory and torch.device(args.device).type != "cuda":
        p.error("--profile-memory requires a CUDA device")
    if args.resume_checkpoint and args.warm_start_checkpoint:p.error("choose resume or warm start, not both")
    if args.stage.endswith("ppo") and not (args.warm_start_checkpoint or args.resume_checkpoint):
        p.error("PPO requires --warm-start-checkpoint from the same stage's supervised run")
    if args.steps<1 or args.epochs<1:p.error("positive steps and epochs required")
    root=Path(args.output_dir)
    if root.exists() and any(root.iterdir()):raise FileExistsError("use a new run directory; base models and historical runs are never overwritten")
    rows=read_training_manifest(args.manifest,args.stage)
    family="audio" if args.stage.startswith("audio") else "symbolic"
    from .paper_runtime import PaperRuntime, discount_per_second
    if not .2 <= args.audio_window_seconds <= .5:
        p.error("paper acoustic windows must be 200--500 ms")
    runtime = None
    if args.stage.endswith("ppo") and args.reward_backend == "paper":
        runtime = PaperRuntime(args.stage,args.paper_config,disable_consistency=args.disable_consistency)
        runtime.preflight(rows)
    unit_discount = args.discount_per_unit if args.discount_per_unit is not None else (.98 if family == "audio" else .95)
    unit_seconds = args.discount_unit_seconds if args.discount_unit_seconds is not None else (args.audio_window_seconds if family == "audio" else float(rows[0].get("duration_seconds", 1.)))
    config=PPOConfig(learning_rate=args.learning_rate,epochs=args.ppo_epochs,clip=args.ppo_clip,value_clip=args.value_clip,
        weight_decay=.01 if args.stage.endswith("sft") else 0.,
        gamma_per_second=discount_per_second(unit_discount,unit_seconds),
        gae_lambda_per_second=discount_per_second(args.gae_lambda,unit_seconds),entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,max_grad_norm=args.max_grad_norm,target_kl=args.target_kl)
    root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    from model_registry import resolve_model
    from experiments.provenance import checkpoint_fingerprint,sha256_file,source_fingerprint
    base=resolve_model("musicgen" if family=="audio" else "mupt",args.model_dir)
    if family=="symbolic":
        from track.melody.infer import load_model
        tokenizer,model=load_model(base,args.device);policy=SymbolicTokenPolicy(model)
    else:
        from audio.musicgen_renderer import MusicGenRendererSession
        from audio.temporal_control import TemporalControlAdapter,AdapterConfig
        renderer=MusicGenRendererSession.get(model_dir=base,device=args.device)
        adapter=(TemporalControlAdapter.load(args.adapter_checkpoint,args.device,expected_base_model=base) if args.adapter_checkpoint
                 else TemporalControlAdapter(AdapterConfig(renderer.model.decoder.config.hidden_size)).to(args.device))
        policy=AudioTokenPolicy(renderer.model,adapter)
    trainer=TokenPPOTrainer(policy,config)
    if args.warm_start_checkpoint or args.resume_checkpoint:
        old=trainer.load(args.resume_checkpoint or args.warm_start_checkpoint,restore_optimizer=bool(args.resume_checkpoint))
        if old.get("family")!=family or old.get("base_model",{}).get("path")!=base:
            raise ValueError("warm start belongs to a different model/stage")
        expected_stage=args.stage if args.resume_checkpoint else family+"-sft"
        if old.get("stage")!=expected_stage:raise ValueError(f"checkpoint stage must be {expected_stage}")
        if args.resume_checkpoint and old.get("reward_backend") != args.reward_backend:
            raise ValueError("resume cannot change reward backend")
        if runtime and args.resume_checkpoint:
            previous=Path(args.resume_checkpoint).parent/"weight_state.json"
            runtime.controller.load_state_dict(json.loads(previous.read_text()))
    metadata={"family":family,"stage":args.stage,"base_model":checkpoint_fingerprint(base,not args.skip_weight_hash),
              "manifest_sha256":sha256_file(args.manifest),"sample_ids":[r["sample_id"] for r in rows],"seed":args.seed,
              "consistency_enabled":not args.disable_consistency,"symbolic_credit":"phrase-terminal; no artificial per-note token timings",
              "audio_credit":f"{args.audio_window_seconds}s acoustic windows assigned by decoder-clock overlap; delayed tail has zero elapsed time",
              "sampling":"categorical temperature=1 top_k=0 top_p=1 CFG=1; matches PPO likelihood", "batch_size":1,
              "trainable_parameters":sum(p.numel() for p in policy.parameters() if p.requires_grad),"config":vars(args)}
    metadata.update(reward_backend=args.reward_backend,
                    source=source_fingerprint(Path(__file__).resolve().parents[1]),
                    discount={"per_unit":unit_discount,"reference_seconds":unit_seconds,
                              "symbolic_unit":"one generated sequence treated as a phrase; no fabricated token-note alignment",
                              "audio_unit":"acoustic window", "gae_lambda_per_unit":args.gae_lambda},
                    paper_runtime=runtime.metadata() if runtime else None)
    from dataclasses import asdict
    metadata["effective_ppo_config"]=asdict(trainer.config)
    metadata["optimizer_initialization"]="resume_checkpoint" if args.resume_checkpoint else "fresh_AdamW"
    metadata["resume_semantics"]="parameter and optimizer continuation; seed and manifest iteration restart, not bitwise mid-epoch replay"
    (root/"training_protocol.json").write_text(json.dumps(metadata,indent=2))
    memory = None
    if args.profile_memory:
        from .memory_profile import TrainingMemoryProfile
        memory = TrainingMemoryProfile(root, args.device, metadata)
    steps=0
    for epoch in range(args.epochs):
        for row in rows:
            if memory:memory.begin_step(steps,row["sample_id"])
            if family=="symbolic" and args.discount_unit_seconds is None:
                reference_seconds=float(row.get("duration_seconds",1.))
                trainer.config.gamma_per_second=discount_per_second(unit_discount,reference_seconds)
                trainer.config.gae_lambda_per_second=discount_per_second(args.gae_lambda,reference_seconds)
            if family=="symbolic":payload,prompt=symbolic_example(row,tokenizer,args.device,supervised=args.stage.endswith("sft"))
            else:payload,blueprint,symbolic=audio_example(row,renderer.model,renderer.processor,args.device,
                                                       supervised=args.stage.endswith("sft"))
            if args.stage.endswith("sft"):
                stats=trainer.warm_start(payload)
            else:
                if family=="symbolic":
                    from symbolic.events import abc_to_events
                    sampled=policy.sample(prompt,args.max_new_tokens,tokenizer.eos_token_id)
                    if memory:memory.mark("after_rollout",sequence_tokens=sampled["tokens"].shape[1],action_tokens=int(sampled["action_mask"].sum()))
                    text=tokenizer.decode(sampled["tokens"][0],skip_special_tokens=True)
                    (root/f"sample_{steps:05d}.abc").write_text(text.replace("<n>","\n"))
                    try:
                        events=abc_to_events(text);pitches=[e.pitch for e in events]
                    except Exception as exc:
                        events=[];pitches=[]
                        score=0.
                        if runtime:
                            runtime.history.append({"step":steps,"sample_id":row["sample_id"],"status":"invalid_score",
                                "error":str(exc),"reward":0.,"invalid_sample_policy":"zero reward; retained in denominator"})
                    else:
                        if runtime and events:
                            score=runtime.symbolic(events,row,float(row.get("duration_seconds",1.)),steps)
                        else:
                            score=(.5*min(1,len(set(pitches))/7)+.5*np.mean([21<=p<=108 for p in pitches])) if pitches else 0.
                            if runtime:
                                runtime.history.append({"step":steps,"sample_id":row["sample_id"],"status":"no_notes","reward":0.})
                    mask=sampled["action_mask"]
                    rewards=torch.zeros(mask.shape,device=args.device);elapsed=torch.zeros_like(rewards)
                    last=int(mask[0].nonzero()[-1]);rewards[0,last]=float(score);elapsed[0,last]=float(row.get("duration_seconds",1.))
                else:
                    import soundfile as sf
                    from reward.artifact_scoring import score_artifacts
                    frames=payload["features"].shape[1]
                    sampled=policy.sample(payload["text_inputs"],payload["features"],frames)
                    if memory:memory.mark("after_rollout",audio_frames=frames,duration_seconds=frames/50)
                    audio=policy.decode(sampled)[0,0].cpu().float().numpy()
                    path=root/f"sample_{steps:05d}.wav";sf.write(path,audio,renderer.sample_rate)
                    mask=sampled["codebook_mask"].reshape(1,renderer.model.decoder.num_codebooks,-1).any(1)
                    times=torch.arange(mask.shape[1],device=args.device)/50
                    duration=frames/50
                    spans=torch.stack((times.clamp_max(duration),(times+.02).clamp_max(duration)),-1)[None]
                    windows=[]
                    # Global components are spread by physical coverage; local onset term
                    # is recomputed per 0.4s window, rather than reusing a whole-clip score.
                    if runtime:
                        rewards,elapsed=runtime.audio([e for item in symbolic for e in item["events"]],row,path,duration,steps,
                            steps/max(1,args.steps-1),mask,args.audio_window_seconds)
                    else:
                        from reward.evidence import extract_audio_features
                        from scipy.signal import find_peaks
                        from metrics.protocol import onset_f1
                        f=extract_audio_features(str(path));peaks,_=find_peaks(f["onset"],height=.2)
                        detected=f["frame_times"][peaks]
                        score_onsets=sorted({e["onset"] for item in symbolic for e in item["events"]})
                        scores=score_artifacts(blueprint,symbolic,{"soundtrack_path":str(path)},consistency_enabled=not args.disable_consistency)
                    for start in ([] if runtime else np.arange(0,duration,args.audio_window_seconds)):
                        end=min(duration,start+args.audio_window_seconds)
                        local=onset_f1([t for t in score_onsets if start<=t<end],[t for t in detected if start<=t<end])
                        base_reward=.5*(scores["rewards"]["R_vis"]+scores["rewards"]["R_mus"])
                        weight=0 if args.disable_consistency else .4+.2*min(1,steps/max(1,args.steps-1))
                        reward=((1-weight)*base_reward+weight*local)*(end-start)/duration
                        windows.append((start,end,reward))
                    if not runtime:
                        rewards=temporal_credit([windows],spans,mask).to(args.device)
                        elapsed=(spans[...,1]-spans[...,0]).to(args.device)
                if memory:memory.mark("after_scoring")
                rollout=trainer.record(sampled,rewards,elapsed)
                if memory:memory.mark("after_old_statistics")
                stats=trainer.update([rollout])
            if memory:memory.mark("after_update",optimizer_updates=len(stats["updates"]) if "updates" in stats else 1)
            record={"step":steps,"epoch":epoch,"sample_id":row["sample_id"],"stats":stats}
            if runtime:
                runtime.flush(root)
                record["reward_trace_index"]=len(runtime.history)-1
            record["discount"]={"gamma_per_second":trainer.config.gamma_per_second,
                                "gae_lambda_per_second":trainer.config.gae_lambda_per_second}
            with (root/"training.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
            print(json.dumps(record),flush=True);steps+=1
            if steps>=args.steps:break
        if steps>=args.steps:break
    trainer.save(root/"policy.pt",metadata)
    if family=="audio":policy.adapter.save(root/"temporal_adapter.pt",metadata=metadata)
    (root/"completion.json").write_text(json.dumps({"steps":steps,"stop":"step_budget" if steps>=args.steps else "epoch_budget","status":"completed"}))
    if memory:memory.finish()
    return 0


if __name__=="__main__":raise SystemExit(main())
