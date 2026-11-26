"""Matched-seed, equal-call-budget refinement comparison with real model outputs."""
import argparse
import json
import time
import platform
from pathlib import Path
from planning import load_blueprint_json
from planning.timeline import merge_overlapping_segments
from reward.refinement import GenerationBudget,refine_inference
from .realization import RealizationBackend
from .provenance import sha256_file,source_fingerprint


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blueprint",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--strategies",nargs="+",default=["none","best_of_n","random","full","routed"],choices=["none","best_of_n","random","full","routed"])
    p.add_argument("--seeds",nargs="+",type=int,default=[123]);p.add_argument("--max-generation-calls",type=int)
    p.add_argument("--max-seconds",type=float);p.add_argument("--max-iterations",type=int,default=5)
    p.add_argument("--proposal-mode",choices=["adaptive","refine-5","refine-30","convergence"],default="adaptive",
                   help="fixed proposal modes have no call cap unless explicitly supplied")
    p.add_argument("--mupt-model-dir",default="");p.add_argument("--musicgen-model-dir",default="")
    p.add_argument("--mupt-device",default="cpu");p.add_argument("--musicgen-device",default="cpu")
    p.add_argument("--adapter-checkpoint",default="");p.add_argument("--disable-consistency",action="store_true")
    p.add_argument("--symbolic-policy-checkpoint",default="")
    p.add_argument("--symbolic-max-new-tokens",type=int,default=384)
    p.add_argument("--local-audio-window-seconds",type=float,default=0.)
    p.add_argument("--paper-reward-inputs", default="", help="measured video and annotated VA JSON; enables six-component scoring")
    p.add_argument("--paper-config", default="configs/paper_training_full.json")
    p.add_argument("--composer-prior-json", default="", help="observed CineParse windows for real LLM blueprint repair")
    args=p.parse_args(argv);root=Path(args.output_dir)
    mode_settings = {"refine-5": (5, "fixed"), "refine-30": (30, "fixed"), "convergence": (60, "convergence")}
    iterations, stopping = mode_settings.get(args.proposal_mode, (args.max_iterations, "adaptive"))
    scorer, planner = None, None
    if args.paper_reward_inputs:
        if args.disable_consistency:
            p.error("six-component paper refinement cannot disable consistency")
        if not args.composer_prior_json:
            p.error("paper refinement requires --composer-prior-json for blueprint proposals")
        from reward.paper_scoring import PaperArtifactScorer
        from main import BlueprintComposerController
        scorer = PaperArtifactScorer(args.paper_reward_inputs, args.paper_config)
        prior = json.loads(Path(args.composer_prior_json).read_text())
        class Planner:
            manages_budget = True
            def __call__(self, current, seed, budget):
                composer = BlueprintComposerController()
                try:
                    return composer.plan_blueprint(prior, budget=budget,
                        revision={"blueprint": current.blueprint.to_dict(), "measured_rewards": current.rewards})
                finally:
                    trace_dir = Path(current.artifact_dir)
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    (trace_dir/f"composer_{budget.calls:06d}_{seed}.json").write_text(json.dumps(getattr(composer, "last_trace", {}), indent=2))
        planner = Planner()
    if root.exists() and any(root.iterdir()):raise FileExistsError("use a new output directory")
    root.mkdir(parents=True,exist_ok=True)
    blueprint=merge_overlapping_segments(load_blueprint_json(args.blueprint))
    from model_registry import resolve_model
    mupt=resolve_model("mupt",args.mupt_model_dir);musicgen=resolve_model("musicgen",args.musicgen_model_dir)
    # Loading occurs once outside each timed trial for a fair warm-cache comparison.
    from track.melody.infer import load_model
    from audio.musicgen_renderer import MusicGenRendererSession
    load_started=time.monotonic()
    load_model(mupt,args.mupt_device,policy_checkpoint=args.symbolic_policy_checkpoint);MusicGenRendererSession.get(model_dir=musicgen,device=args.musicgen_device)
    import torch
    hardware={"platform":platform.platform(),"torch":str(torch.__version__),"batch_size":1,
              "cuda_devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "model_load_seconds":time.monotonic()-load_started,
              "llm_api_timing":"included in budget_stages when Composer enabled"}
    records=[]
    for seed in args.seeds:
        for strategy in args.strategies:
            run_dir=root/f"seed_{seed}"/strategy
            backend=RealizationBackend(str(run_dir),mupt,musicgen,args.mupt_device,args.musicgen_device,args.adapter_checkpoint,
                                       not args.disable_consistency,independent_sampling=strategy=="best_of_n",
                symbolic_policy_checkpoint=args.symbolic_policy_checkpoint,max_new_tokens=args.symbolic_max_new_tokens,
                local_audio_window_seconds=args.local_audio_window_seconds,scorer=scorer,planner=planner)
            call_limit = args.max_generation_calls
            if call_limit is None and args.proposal_mode == "adaptive":
                call_limit = 64
            budget=GenerationBudget(call_limit,args.max_seconds)
            try:
                initial=backend.generate(blueprint,seed=seed,budget=budget)
                initial_calls=budget.calls
                final,trace=refine_inference(initial,backend.regenerate,budget=budget,strategy=strategy,max_iterations=iterations,
                                            stopping=stopping,patience=5 if stopping=="convergence" else 2,seed=seed)
                record={"seed":seed,"strategy":strategy,"status":"ok","initial_calls":initial_calls,
                        "initial_rewards":initial.rewards,"final_rewards":final.rewards,"final_audio":final.audio_manifest,
                        "final_artifact_dir":final.artifact_dir,"trace":trace}
            except Exception as exc:
                record={"seed":seed,"strategy":strategy,"status":"failed","error":str(exc),"calls":budget.calls,
                        "elapsed_seconds":time.monotonic()-budget.started}
            records.append(record)
            (root/"results.json").write_text(json.dumps({"protocol":vars(args),"records":records,
                "blueprint_sha256":sha256_file(args.blueprint),"base_models":{"mupt":mupt,"musicgen":musicgen},
                "hardware":hardware,
                "scope":"six-component frozen paper scoring with LLM Composer" if scorer else "artifact proxy scoring with deterministic blueprint repair",
                "timing":"warm models; generation, retries, artifact IO and scoring included; atomic in-flight calls may exceed deadline",
                "source":source_fingerprint(Path(__file__).resolve().parents[1])},indent=2))
    return 0 if all(r["status"]=="ok" for r in records) else 1

if __name__=="__main__":raise SystemExit(main())
