from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_controls_to_prompts
from audio.musicgen_renderer import render_musicgen_prompts
from audio.reference_audio import extract_reference_audio_features, reference_audio_for_llm
from audio.symbolic_renderer import generate_symbolic_abc, generate_blueprint_score
from metrics import evaluate_paper_metrics
from planning import (
    MusicBlueprint,
    blueprint_to_bar_contexts,
    blueprint_to_mupt_prefixes,
    load_blueprint_json,
    save_blueprint_json,
)
from reward.config import build_default_rl_config
from reward.evidence import ModelBackedRewardEvaluator, RewardEvidenceEvaluator
from reward.refine import refine_blueprint_controls, refine_blueprint_hierarchical
from vedio.prior_feature_pack import extract_prior_5d


def _write_lines(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _prefix_segment_ids(blueprint: MusicBlueprint) -> List[int]:
    segment_ids: List[int] = []
    for segment in blueprint.segments:
        segment_ids.extend([segment.segment_id] * max(1, len(segment.chords)))
    return segment_ids


def _build_rl_config(
    *,
    critic_feedback_mode: str,
    human_feedback: dict[str, Any] | None,
    llm_feedback: dict[str, Any] | None,
    reward_backend: str = "auto",
    symbolic_warmup_episodes: int = 20,
    update_symbolic_during_joint: bool = False,
    llm_critic_base_url: str = "",
    llm_critic_api_key: str = "",
    llm_critic_model: str = "",
):
    rl_config = build_default_rl_config()
    rl_config.critic_feedback_mode = critic_feedback_mode
    rl_config.human_feedback_enabled = human_feedback is not None
    rl_config.use_llm_controller = "llm" in critic_feedback_mode or llm_feedback is not None
    rl_config.reward_backend = reward_backend
    rl_config.symbolic_warmup_episodes = max(0, symbolic_warmup_episodes)
    rl_config.update_symbolic_during_joint = update_symbolic_during_joint
    if llm_critic_model:
        rl_config.llm_critic_enabled = True
        if llm_critic_base_url:
            rl_config.llm_critic_base_url = llm_critic_base_url
        if llm_critic_api_key:
            rl_config.llm_critic_api_key = llm_critic_api_key
        rl_config.llm_critic_model = llm_critic_model
    return rl_config


def _write_repair_artifacts(out_dir: Path, repair_output: dict[str, Any], suffix: str) -> None:
    _write_json(out_dir / f"repair_cycle{suffix}.json", repair_output)
    _write_json(out_dir / f"critic_decision_repaired{suffix}.json", repair_output["critic_decision"])
    _write_json(out_dir / f"blueprint_repaired{suffix}.json", repair_output["repaired_blueprint"])
    _write_json(out_dir / f"bar_contexts_repaired{suffix}.json", repair_output["repaired_bar_contexts"])
    _write_lines(repair_output["repaired_mupt_prefixes"], out_dir / f"mupt_prefixes_repaired{suffix}.txt")
    _write_json(out_dir / f"render_controls_repaired{suffix}.json", repair_output["repaired_render_controls"])
    _write_lines(repair_output["repaired_prompts"], out_dir / f"musicgen_prompts_repaired{suffix}.txt")


def run_pipeline(
    *,
    video_path: str,
    output_dir: str,
    segment_duration: float,
    segment_overlap: float,
    sample_fps: float,
    cut_threshold: float,
    refine_with_rl: bool,
    refine_with_hierarchical_rl: bool,
    rl_episodes: int,
    enable_repair_cycle: bool,
    critic_feedback_mode: str,
    human_feedback: dict[str, Any] | None,
    llm_feedback: dict[str, Any] | None,
    user_prompt: str = "",
    reference_hint: str = "",
    reference_audio_path: str = "",
    generate_symbolic: bool = False,
    generate_audio: bool = False,
    mupt_model_dir: str = "",
    mupt_device: str = "",
    musicgen_model_dir: str = "",
    musicgen_device: str = "",
    rl_reward_mode: str = "auto",
    symbolic_warmup_episodes: int = 20,
    update_symbolic_during_joint: bool = False,
    llm_critic_base_url: str = "",
    llm_critic_api_key: str = "",
    llm_critic_model: str = "",
    evaluate_metrics: bool = False,
    fad_reference_dir: str = "",
    imagebind_video_embedding: str = "",
    imagebind_audio_embedding: str = "",
    inception_probabilities: str = "",
    clap_model_dir: str = "",
    seed: int = 123,
    adapter_checkpoint: str = "",
    symbolic_policy_checkpoint: str = "",
    symbolic_max_new_tokens: int = 384,
    metric_inputs: dict | None = None,
    reconstruction_reference_audio_path: str = "",
    inference_strategy: str = "routed",
    inference_max_calls: int = 64,
    inference_iterations: int = 5,
    local_audio_window_seconds: float = 0.,
    cineparse_checkpoint: str = "",
    cineparse_backbone_checkpoint: str = "",
    cineparse_device: str = "cpu",
) -> None:
    prior = extract_prior_5d(
        video_path,
        segment_duration=segment_duration,
        segment_overlap=segment_overlap,
        sample_fps=sample_fps,
        cut_threshold=cut_threshold,
        cineparse_checkpoint=cineparse_checkpoint,
        cineparse_backbone_checkpoint=cineparse_backbone_checkpoint,
        cineparse_device=cineparse_device,
    )
    run_pipeline_from_prior(
        prior=prior,
        output_dir=output_dir,
        refine_with_rl=refine_with_rl,
        refine_with_hierarchical_rl=refine_with_hierarchical_rl,
        rl_episodes=rl_episodes,
        enable_repair_cycle=enable_repair_cycle,
        critic_feedback_mode=critic_feedback_mode,
        human_feedback=human_feedback,
        llm_feedback=llm_feedback,
        video_path=video_path,
        user_prompt=user_prompt,
        reference_hint=reference_hint,
        reference_audio_path=reference_audio_path,
        generate_symbolic=generate_symbolic,
        generate_audio=generate_audio,
        mupt_model_dir=mupt_model_dir,
        mupt_device=mupt_device,
        musicgen_model_dir=musicgen_model_dir,
        musicgen_device=musicgen_device,
        rl_reward_mode=rl_reward_mode,
        symbolic_warmup_episodes=symbolic_warmup_episodes,
        update_symbolic_during_joint=update_symbolic_during_joint,
        llm_critic_base_url=llm_critic_base_url,
        llm_critic_api_key=llm_critic_api_key,
        llm_critic_model=llm_critic_model,
        evaluate_metrics=evaluate_metrics,
        fad_reference_dir=fad_reference_dir,
        imagebind_video_embedding=imagebind_video_embedding,
        imagebind_audio_embedding=imagebind_audio_embedding,
        inception_probabilities=inception_probabilities,
        clap_model_dir=clap_model_dir,
        seed=seed, adapter_checkpoint=adapter_checkpoint,
        symbolic_policy_checkpoint=symbolic_policy_checkpoint,
        symbolic_max_new_tokens=symbolic_max_new_tokens, metric_inputs=metric_inputs,
        reconstruction_reference_audio_path=reconstruction_reference_audio_path,
        inference_strategy=inference_strategy, inference_max_calls=inference_max_calls,
        inference_iterations=inference_iterations,
        local_audio_window_seconds=local_audio_window_seconds,
    )


def run_pipeline_from_prior(
    *,
    prior: dict[str, Any],
    output_dir: str,
    refine_with_rl: bool,
    refine_with_hierarchical_rl: bool,
    rl_episodes: int,
    enable_repair_cycle: bool,
    critic_feedback_mode: str,
    human_feedback: dict[str, Any] | None,
    llm_feedback: dict[str, Any] | None,
    video_path: str = "",
    user_prompt: str = "",
    reference_hint: str = "",
    reference_audio_path: str = "",
    generate_symbolic: bool = False,
    generate_audio: bool = False,
    mupt_model_dir: str = "",
    mupt_device: str = "",
    musicgen_model_dir: str = "",
    musicgen_device: str = "",
    rl_reward_mode: str = "auto",
    symbolic_warmup_episodes: int = 20,
    update_symbolic_during_joint: bool = False,
    llm_critic_base_url: str = "",
    llm_critic_api_key: str = "",
    llm_critic_model: str = "",
    evaluate_metrics: bool = False,
    fad_reference_dir: str = "",
    imagebind_video_embedding: str = "",
    imagebind_audio_embedding: str = "",
    inception_probabilities: str = "",
    clap_model_dir: str = "",
    seed: int = 123,
    adapter_checkpoint: str = "",
    symbolic_policy_checkpoint: str = "",
    symbolic_max_new_tokens: int = 384,
    metric_inputs: dict | None = None,
    reconstruction_reference_audio_path: str = "",
    inference_strategy: str = "routed",
    inference_max_calls: int = 64,
    inference_iterations: int = 5,
    local_audio_window_seconds: float = 0.,
) -> None:
    from main import BlueprintComposerController

    destination=Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("use a new output directory; existing artifacts are preserved")

    reference_audio_features = (
        extract_reference_audio_features(reference_audio_path)
        if reference_audio_path
        else None
    )
    composer = BlueprintComposerController()
    blueprint = composer.plan_blueprint(
        prior,
        user_prompt=user_prompt,
        reference_hint=reference_hint,
        reference_audio=reference_audio_features,
    )
    _write_json(Path(output_dir) / "composer_trace.json", composer.last_trace)
    run_pipeline_from_blueprint(
        blueprint=blueprint,
        output_dir=output_dir,
        refine_with_rl=refine_with_rl,
        refine_with_hierarchical_rl=refine_with_hierarchical_rl,
        rl_episodes=rl_episodes,
        enable_repair_cycle=enable_repair_cycle,
        critic_feedback_mode=critic_feedback_mode,
        human_feedback=human_feedback,
        llm_feedback=llm_feedback,
        video_path=video_path,
        prior=prior,
        reference_audio_features=reference_audio_features,
        input_mode="video",
        generate_symbolic=generate_symbolic,
        generate_audio=generate_audio,
        mupt_model_dir=mupt_model_dir,
        mupt_device=mupt_device,
        musicgen_model_dir=musicgen_model_dir,
        musicgen_device=musicgen_device,
        rl_reward_mode=rl_reward_mode,
        symbolic_warmup_episodes=symbolic_warmup_episodes,
        update_symbolic_during_joint=update_symbolic_during_joint,
        llm_critic_base_url=llm_critic_base_url,
        llm_critic_api_key=llm_critic_api_key,
        llm_critic_model=llm_critic_model,
        user_prompt=user_prompt,
        reference_audio_path=reference_audio_path,
        evaluate_metrics=evaluate_metrics,
        fad_reference_dir=fad_reference_dir,
        imagebind_video_embedding=imagebind_video_embedding,
        imagebind_audio_embedding=imagebind_audio_embedding,
        inception_probabilities=inception_probabilities,
        clap_model_dir=clap_model_dir,
        seed=seed, adapter_checkpoint=adapter_checkpoint,
        symbolic_policy_checkpoint=symbolic_policy_checkpoint,
        symbolic_max_new_tokens=symbolic_max_new_tokens, metric_inputs=metric_inputs,
        reconstruction_reference_audio_path=reconstruction_reference_audio_path,
        inference_strategy=inference_strategy, inference_max_calls=inference_max_calls,
        inference_iterations=inference_iterations,
        local_audio_window_seconds=local_audio_window_seconds,
    )


def run_pipeline_from_blueprint(
    *,
    blueprint: MusicBlueprint,
    output_dir: str,
    refine_with_rl: bool,
    refine_with_hierarchical_rl: bool,
    rl_episodes: int,
    enable_repair_cycle: bool,
    critic_feedback_mode: str,
    human_feedback: dict[str, Any] | None,
    llm_feedback: dict[str, Any] | None,
    video_path: str = "",
    prior: dict[str, Any] | None = None,
    reference_audio_features: dict[str, Any] | None = None,
    input_mode: str = "blueprint",
    generate_symbolic: bool = False,
    generate_audio: bool = False,
    mupt_model_dir: str = "",
    mupt_device: str = "",
    musicgen_model_dir: str = "",
    musicgen_device: str = "",
    rl_reward_mode: str = "auto",
    symbolic_warmup_episodes: int = 20,
    update_symbolic_during_joint: bool = False,
    llm_critic_base_url: str = "",
    llm_critic_api_key: str = "",
    llm_critic_model: str = "",
    user_prompt: str = "",
    reference_audio_path: str = "",
    evaluate_metrics: bool = False,
    fad_reference_dir: str = "",
    imagebind_video_embedding: str = "",
    imagebind_audio_embedding: str = "",
    inception_probabilities: str = "",
    clap_model_dir: str = "",
    seed: int = 123,
    adapter_checkpoint: str = "",
    symbolic_policy_checkpoint: str = "",
    symbolic_max_new_tokens: int = 384,
    metric_inputs: dict | None = None,
    reconstruction_reference_audio_path: str = "",
    inference_strategy: str = "routed",
    inference_max_calls: int = 64,
    inference_iterations: int = 5,
    local_audio_window_seconds: float = 0.,
) -> None:
    import time, random, numpy as np, torch
    started_at = time.monotonic()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from planning.timeline import merge_overlapping_segments
    out_dir = Path(output_dir)
    if out_dir.exists() and any(p.name!="composer_trace.json" for p in out_dir.iterdir()):
        raise FileExistsError("use a new output directory; existing artifacts are preserved")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_blueprint_json(blueprint, str(out_dir / "blueprint_observation_windows.json"))
    blueprint = merge_overlapping_segments(blueprint)
    from model_registry import resolve_model
    if generate_symbolic or rl_reward_mode == "model":
        mupt_model_dir = resolve_model("mupt", mupt_model_dir)
    if generate_audio or rl_reward_mode == "model":
        musicgen_model_dir = resolve_model("musicgen", musicgen_model_dir)
    rl_config = _build_rl_config(
        critic_feedback_mode=critic_feedback_mode,
        human_feedback=human_feedback,
        llm_feedback=llm_feedback,
        reward_backend=rl_reward_mode,
        symbolic_warmup_episodes=symbolic_warmup_episodes,
        update_symbolic_during_joint=update_symbolic_during_joint,
        llm_critic_base_url=llm_critic_base_url,
        llm_critic_api_key=llm_critic_api_key,
        llm_critic_model=llm_critic_model,
    )

    if prior is not None:
        _write_json(out_dir / "prior.json", prior)
    if reference_audio_features is None and reference_audio_path:
        reference_audio_features = extract_reference_audio_features(reference_audio_path)
    if reference_audio_features is not None:
        _write_json(out_dir / "reference_audio_features.json", reference_audio_for_llm(reference_audio_features))

    save_blueprint_json(blueprint, str(out_dir / "blueprint.json"))

    contexts = blueprint_to_bar_contexts(blueprint)
    _write_json(out_dir / "bar_contexts.json", contexts)

    prefixes = blueprint_to_mupt_prefixes(blueprint)
    _write_lines(prefixes, out_dir / "mupt_prefixes.txt")

    symbolic_outputs: List[dict[str, object]] = []
    if generate_symbolic:
        symbolic_outputs = generate_blueprint_score(
            blueprint, output_dir=str(out_dir / "symbolic"),
            model_dir=mupt_model_dir or None, device=mupt_device or None, seed=seed,
            policy_checkpoint=symbolic_policy_checkpoint,max_new_tokens=symbolic_max_new_tokens,
        )
        _write_json(out_dir / "symbolic_outputs.json", symbolic_outputs)

    render_controls = blueprint_to_render_controls(blueprint, symbolic_outputs=symbolic_outputs)
    _write_json(out_dir / "render_controls.json", render_controls)
    musicgen_prompts = render_controls_to_prompts(render_controls)
    _write_lines(musicgen_prompts, out_dir / "musicgen_prompts.txt")

    audio_manifest: dict[str, object] = {}
    if generate_audio:
        audio_manifest = render_musicgen_prompts(
            musicgen_prompts,
            model_dir=musicgen_model_dir,
            output_dir=str(out_dir / "audio"),
            device=musicgen_device or None,
            controls=render_controls, symbolic_outputs=symbolic_outputs,
            seed=seed, adapter_checkpoint=adapter_checkpoint,
        )

    if rl_reward_mode == "model":
        if not mupt_model_dir or not musicgen_model_dir:
            raise ValueError("model reward mode requires both --mupt-model-dir and --musicgen-model-dir")
        reward_evaluator = ModelBackedRewardEvaluator(
            output_dir=str(out_dir / "rl_candidates"),
            mupt_model_dir=mupt_model_dir,
            musicgen_model_dir=musicgen_model_dir,
            mupt_device=mupt_device or None,
            musicgen_device=musicgen_device or None,
            temporal_alignment_freq=rl_config.temporal_alignment_freq,
        )
    else:
        reward_evaluator = RewardEvidenceEvaluator(
            symbolic_outputs=symbolic_outputs,
            audio_manifest=audio_manifest,
            temporal_alignment_freq=rl_config.temporal_alignment_freq,
        )

    refinement = None
    if refine_with_hierarchical_rl:
        refinement = refine_blueprint_hierarchical(
            blueprint,
            config=rl_config,
            num_episodes=rl_episodes,
            human_feedback=human_feedback,
            llm_feedback=llm_feedback,
            reward_evaluator=reward_evaluator,
        )
        _write_json(out_dir / "rl_refinement_hierarchical.json", refinement)
        _write_json(out_dir / "critic_decision_hierarchical.json", refinement["critic_decision"])
        _write_json(out_dir / "blueprint_refined_hierarchical.json", refinement["refined_blueprint"])
        _write_json(out_dir / "bar_contexts_refined_hierarchical.json", refinement["refined_bar_contexts"])
        _write_lines(refinement["refined_mupt_prefixes"], out_dir / "mupt_prefixes_refined_hierarchical.txt")
        _write_json(out_dir / "render_controls_refined_hierarchical.json", refinement["refined_controls"])
        _write_lines(refinement["refined_prompts"], out_dir / "musicgen_prompts_refined_hierarchical.txt")
        if enable_repair_cycle:
            _write_repair_artifacts(out_dir, refinement["repair_output"], "_hierarchical")
    elif refine_with_rl:
        refinement = refine_blueprint_controls(
            blueprint,
            config=rl_config,
            num_episodes=rl_episodes,
            human_feedback=human_feedback,
            llm_feedback=llm_feedback,
            reward_evaluator=reward_evaluator,
        )
        _write_json(out_dir / "rl_refinement.json", refinement)
        _write_json(out_dir / "critic_decision.json", refinement["critic_decision"])
        _write_json(out_dir / "render_controls_refined.json", refinement["refined_controls"])
        _write_lines(refinement["refined_prompts"], out_dir / "musicgen_prompts_refined.txt")
        if enable_repair_cycle:
            _write_repair_artifacts(out_dir, refinement["repair_output"], "")

    # Realize the controls chosen by legacy control-PPO before final evaluation.
    # Its separate optimization target is recorded in the manifest.
    if refinement is not None and (generate_symbolic or generate_audio):
        from planning.blueprint_parser import MusicBlueprintParser
        if "refined_blueprint" in refinement:
            blueprint = MusicBlueprintParser().parse(refinement["refined_blueprint"],source="control_ppo")
        render_controls = refinement["refined_controls"]
        if generate_symbolic:
            symbolic_outputs = generate_blueprint_score(blueprint,output_dir=str(out_dir/"refined_symbolic"),
                model_dir=mupt_model_dir,device=mupt_device or None,seed=seed,
                policy_checkpoint=symbolic_policy_checkpoint,max_new_tokens=symbolic_max_new_tokens)
        if generate_audio:
            audio_manifest = render_musicgen_prompts(render_controls_to_prompts(render_controls),model_dir=musicgen_model_dir,
                output_dir=str(out_dir/"refined_audio"),device=musicgen_device or None,controls=render_controls,
                symbolic_outputs=symbolic_outputs,adapter_checkpoint=adapter_checkpoint,seed=seed)
    if enable_repair_cycle and symbolic_outputs and audio_manifest:
        from reward.artifact_scoring import score_artifacts
        from reward.refinement import Candidate,GenerationBudget,refine_inference
        from experiments.realization import RealizationBackend
        initial_scores=score_artifacts(blueprint,symbolic_outputs,audio_manifest)
        initial=Candidate(blueprint,symbolic_outputs,audio_manifest,initial_scores["rewards"],str(out_dir))
        backend=RealizationBackend(str(out_dir/"inference_candidates"),mupt_model_dir,musicgen_model_dir,
            mupt_device or "cpu",musicgen_device or "cpu",adapter_checkpoint,
            independent_sampling=inference_strategy=="best_of_n",
            symbolic_policy_checkpoint=symbolic_policy_checkpoint,max_new_tokens=symbolic_max_new_tokens,
            local_audio_window_seconds=local_audio_window_seconds)
        budget=GenerationBudget(inference_max_calls)
        from reward.critic import build_critic_decision
        from reward.llm_critic_client import LLMStructuredCritic
        structured_critic=LLMStructuredCritic.from_config(rl_config)
        def infer_decision(candidate):
            decision=build_critic_decision(candidate.rewards,feedback_mode=critic_feedback_mode,
                global_threshold=rl_config.repair_global_threshold,component_threshold=rl_config.repair_component_threshold,
                max_repair_passes=inference_iterations,human_feedback=human_feedback,llm_feedback=llm_feedback,
                llm_critic=structured_critic,critic_context={"blueprint":candidate.blueprint.to_dict(),
                    "artifact_dir":candidate.artifact_dir,"reward_scope":"artifact_proxies_not_paper_metrics"})
            return {**decision.diagnosis,"feedback_source":decision.feedback_source,
                    "human_veto":decision.feedback_source=="human" and bool((human_feedback or {}).get("veto"))}
        # Here the budget is additional calls. The equal-budget experiment runner
        # includes initial generation in its own total budget.
        final,trace=refine_inference(initial,backend.regenerate,budget=budget,strategy=inference_strategy,
                                    max_iterations=inference_iterations,seed=seed,decision_callback=infer_decision)
        trace["budget_scope"]="additional_calls_after_initial_realization"
        _write_json(out_dir/"inference_refinement.json",trace)
        blueprint,symbolic_outputs,audio_manifest=final.blueprint,final.symbolic_outputs,final.audio_manifest
    elif enable_repair_cycle:
        _write_json(out_dir/"inference_refinement.json",{"status":"not_run","reason":"real symbolic and audio artifacts are required","parameters_updated":False})
    save_blueprint_json(blueprint,str(out_dir/"final_blueprint.json"))
    _write_json(out_dir/"final_symbolic_outputs.json",symbolic_outputs)
    _write_json(out_dir/"final_audio_manifest.json",audio_manifest)
    paper_metric_output: dict[str, Any] = {}
    if evaluate_metrics:
        paper_metric_output = evaluate_paper_metrics(
            blueprint=blueprint,
            symbolic_outputs=symbolic_outputs,
            audio_manifest=audio_manifest,
            prompt=user_prompt,
            reference_audio_path=reference_audio_path,
            fad_reference_dir=fad_reference_dir,
            imagebind_video_embedding=imagebind_video_embedding,
            imagebind_audio_embedding=imagebind_audio_embedding,
            inception_probabilities=inception_probabilities,
            clap_model_dir=clap_model_dir,
            clap_device=musicgen_device or None,
            reconstruction_reference_audio_path=reconstruction_reference_audio_path,
            visual_features=(prior or {}).get("frame_features"),
            metric_inputs=metric_inputs,
        )
        _write_json(out_dir / "paper_metrics.json", paper_metric_output)

    pipeline_log = {
        "input_mode": input_mode,
        "seed": seed,
        "runtime_seconds_including_load": time.monotonic()-started_at,
        "training_backend": "control_mlp_ppo" if refine_with_rl or refine_with_hierarchical_rl else None,
        "adapter_checkpoint": adapter_checkpoint or None,
        "symbolic_policy_checkpoint": symbolic_policy_checkpoint or None,
        "symbolic_max_new_tokens": symbolic_max_new_tokens,
        "source_revision": "source_release",
        "video_path": video_path,
        "num_segments": len(blueprint.segments),
        "num_prefixes": len(prefixes),
        "rl_refinement_enabled": refine_with_rl,
        "hierarchical_rl_refinement_enabled": refine_with_hierarchical_rl,
        "symbolic_generation_enabled": generate_symbolic,
        "audio_generation_enabled": generate_audio,
        "symbolic_outputs": len(symbolic_outputs),
        "audio_manifest": audio_manifest,
        "rl_reward_mode": rl_reward_mode,
        "repair_cycle_enabled": enable_repair_cycle,
        "paper_metrics_enabled": evaluate_metrics,
        "artifacts": {
            "blueprint": str(out_dir / "blueprint.json"),
            "bar_contexts": str(out_dir / "bar_contexts.json"),
            "mupt_prefixes": str(out_dir / "mupt_prefixes.txt"),
            "render_controls": str(out_dir / "render_controls.json"),
            "musicgen_prompts": str(out_dir / "musicgen_prompts.txt"),
        },
    }
    if prior is not None:
        pipeline_log["artifacts"]["prior"] = str(out_dir / "prior.json")
        pipeline_log["visual_extractor"] = prior.get("extractor")
    if reference_audio_features is not None:
        pipeline_log["artifacts"]["reference_audio_features"] = str(out_dir / "reference_audio_features.json")
    if evaluate_metrics:
        pipeline_log["artifacts"]["paper_metrics"] = str(out_dir / "paper_metrics.json")
    if refine_with_hierarchical_rl:
        pipeline_log["artifacts"]["rl_refinement_hierarchical"] = str(out_dir / "rl_refinement_hierarchical.json")
        pipeline_log["artifacts"]["critic_decision_hierarchical"] = str(out_dir / "critic_decision_hierarchical.json")
        pipeline_log["artifacts"]["blueprint_refined_hierarchical"] = str(out_dir / "blueprint_refined_hierarchical.json")
        pipeline_log["artifacts"]["bar_contexts_refined_hierarchical"] = str(out_dir / "bar_contexts_refined_hierarchical.json")
        pipeline_log["artifacts"]["mupt_prefixes_refined_hierarchical"] = str(out_dir / "mupt_prefixes_refined_hierarchical.txt")
        pipeline_log["artifacts"]["render_controls_refined_hierarchical"] = str(out_dir / "render_controls_refined_hierarchical.json")
        pipeline_log["artifacts"]["musicgen_prompts_refined_hierarchical"] = str(out_dir / "musicgen_prompts_refined_hierarchical.txt")
        if enable_repair_cycle:
            pipeline_log["artifacts"]["repair_cycle_hierarchical"] = str(out_dir / "repair_cycle_hierarchical.json")
            pipeline_log["artifacts"]["critic_decision_repaired_hierarchical"] = str(out_dir / "critic_decision_repaired_hierarchical.json")
            pipeline_log["artifacts"]["blueprint_repaired_hierarchical"] = str(out_dir / "blueprint_repaired_hierarchical.json")
            pipeline_log["artifacts"]["bar_contexts_repaired_hierarchical"] = str(out_dir / "bar_contexts_repaired_hierarchical.json")
            pipeline_log["artifacts"]["mupt_prefixes_repaired_hierarchical"] = str(out_dir / "mupt_prefixes_repaired_hierarchical.txt")
            pipeline_log["artifacts"]["render_controls_repaired_hierarchical"] = str(out_dir / "render_controls_repaired_hierarchical.json")
            pipeline_log["artifacts"]["musicgen_prompts_repaired_hierarchical"] = str(out_dir / "musicgen_prompts_repaired_hierarchical.txt")
    elif refine_with_rl:
        pipeline_log["artifacts"]["rl_refinement"] = str(out_dir / "rl_refinement.json")
        pipeline_log["artifacts"]["critic_decision"] = str(out_dir / "critic_decision.json")
        pipeline_log["artifacts"]["render_controls_refined"] = str(out_dir / "render_controls_refined.json")
        pipeline_log["artifacts"]["musicgen_prompts_refined"] = str(out_dir / "musicgen_prompts_refined.txt")
        if enable_repair_cycle:
            pipeline_log["artifacts"]["repair_cycle"] = str(out_dir / "repair_cycle.json")
            pipeline_log["artifacts"]["critic_decision_repaired"] = str(out_dir / "critic_decision_repaired.json")
            pipeline_log["artifacts"]["blueprint_repaired"] = str(out_dir / "blueprint_repaired.json")
            pipeline_log["artifacts"]["bar_contexts_repaired"] = str(out_dir / "bar_contexts_repaired.json")
            pipeline_log["artifacts"]["mupt_prefixes_repaired"] = str(out_dir / "mupt_prefixes_repaired.txt")
            pipeline_log["artifacts"]["render_controls_repaired"] = str(out_dir / "render_controls_repaired.json")
            pipeline_log["artifacts"]["musicgen_prompts_repaired"] = str(out_dir / "musicgen_prompts_repaired.txt")
    from experiments.provenance import canonical_hash,source_fingerprint
    from metrics.protocol import VERSION
    pipeline_log["provenance"]={"blueprint_sha256":canonical_hash(blueprint.to_dict()),
        "source_sha256":source_fingerprint(Path(__file__).resolve().parent)["sha256"],
        "models":{"mupt":mupt_model_dir,"musicgen":musicgen_model_dir},"evaluation_version":VERSION,
        "input_condition":"V"+("+q" if user_prompt else "")+("+r" if reference_audio_path else ""),
        "paper_table_eligible":False,"reason":"single run; dataset/split IDs and complete checkpoint hashes require experiment manifest"}
    pipeline_log["artifacts"].update({"final_blueprint":str(out_dir/"final_blueprint.json"),
        "final_symbolic_outputs":str(out_dir/"final_symbolic_outputs.json"),"final_audio_manifest":str(out_dir/"final_audio_manifest.json")})
    _write_json(out_dir / "pipeline_log.json", pipeline_log)


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_coma_pipeline")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video-path")
    input_group.add_argument("--blueprint-json")
    parser.add_argument("--output-dir", default="output/coma_pipeline")
    parser.add_argument("--segment-duration", type=float, default=2.0)
    parser.add_argument("--segment-overlap", type=float, default=0.5)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--cut-threshold", type=float, default=0.35)
    parser.add_argument("--cineparse-checkpoint", default="")
    parser.add_argument("--cineparse-backbone-checkpoint", default="")
    parser.add_argument("--cineparse-device", default="cpu")
    parser.add_argument("--refine-with-rl", action="store_true")
    parser.add_argument("--refine-with-hierarchical-rl", action="store_true")
    parser.add_argument("--rl-episodes", type=int, default=20)
    parser.add_argument("--enable-repair-cycle", action="store_true")
    parser.add_argument("--critic-feedback-mode", default="llm_or_human")
    parser.add_argument("--human-feedback-json", default="")
    parser.add_argument("--llm-feedback-json", default="")
    parser.add_argument("--user-prompt", default="")
    parser.add_argument("--reference-hint", default="")
    parser.add_argument("--reference-audio", default="")
    parser.add_argument("--generate-symbolic", action="store_true")
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument("--mupt-model-dir", default="")
    parser.add_argument("--mupt-device", default="")
    parser.add_argument("--musicgen-model-dir", default="")
    parser.add_argument("--musicgen-device", default="")
    parser.add_argument("--rl-reward-mode", choices=["auto", "precomputed", "model"], default="auto")
    parser.add_argument("--symbolic-warmup-episodes", type=int, default=20)
    parser.add_argument("--update-symbolic-during-joint", action="store_true")
    parser.add_argument("--llm-critic-base-url", default="")
    parser.add_argument("--llm-critic-api-key", default="")
    parser.add_argument("--llm-critic-model", default="")
    parser.add_argument("--evaluate-paper-metrics", action="store_true")
    parser.add_argument("--fad-reference-dir", default="")
    parser.add_argument("--imagebind-video-embedding", default="")
    parser.add_argument("--imagebind-audio-embedding", default="")
    parser.add_argument("--inception-probabilities", default="")
    parser.add_argument("--clap-model-dir", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--adapter-checkpoint", default="")
    parser.add_argument("--symbolic-policy-checkpoint", default="")
    parser.add_argument("--symbolic-max-new-tokens",type=int,default=384)
    parser.add_argument("--metric-inputs-json",default="")
    parser.add_argument("--reconstruction-reference-audio", default="")
    parser.add_argument("--inference-strategy",choices=["none","best_of_n","random","full","routed"],default="routed")
    parser.add_argument("--inference-max-calls",type=int,default=64)
    parser.add_argument("--inference-iterations",type=int,default=5)
    parser.add_argument("--local-audio-window-seconds",type=float,default=0.)
    args = parser.parse_args()
    if not args.video_path and (args.cineparse_checkpoint or args.cineparse_backbone_checkpoint):
        parser.error("CineParse options require --video-path")

    human_feedback = json.loads(Path(args.human_feedback_json).read_text(encoding="utf-8")) if args.human_feedback_json else None
    llm_feedback = json.loads(Path(args.llm_feedback_json).read_text(encoding="utf-8")) if args.llm_feedback_json else None

    if args.video_path:
        run_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir,
            segment_duration=args.segment_duration,
            segment_overlap=args.segment_overlap,
            sample_fps=args.sample_fps,
            cut_threshold=args.cut_threshold,
            cineparse_checkpoint=args.cineparse_checkpoint,
            cineparse_backbone_checkpoint=args.cineparse_backbone_checkpoint,
            cineparse_device=args.cineparse_device,
            refine_with_rl=args.refine_with_rl,
            refine_with_hierarchical_rl=args.refine_with_hierarchical_rl,
            rl_episodes=args.rl_episodes,
            enable_repair_cycle=args.enable_repair_cycle,
            critic_feedback_mode=args.critic_feedback_mode,
            human_feedback=human_feedback,
            llm_feedback=llm_feedback,
            user_prompt=args.user_prompt,
            reference_hint=args.reference_hint,
            reference_audio_path=args.reference_audio,
            generate_symbolic=args.generate_symbolic,
            generate_audio=args.generate_audio,
            mupt_model_dir=args.mupt_model_dir,
            mupt_device=args.mupt_device,
            musicgen_model_dir=args.musicgen_model_dir,
            musicgen_device=args.musicgen_device,
            rl_reward_mode=args.rl_reward_mode,
            symbolic_warmup_episodes=args.symbolic_warmup_episodes,
            update_symbolic_during_joint=args.update_symbolic_during_joint,
            llm_critic_base_url=args.llm_critic_base_url,
            llm_critic_api_key=args.llm_critic_api_key,
            llm_critic_model=args.llm_critic_model,
            evaluate_metrics=args.evaluate_paper_metrics,
            fad_reference_dir=args.fad_reference_dir,
            imagebind_video_embedding=args.imagebind_video_embedding,
            imagebind_audio_embedding=args.imagebind_audio_embedding,
            inception_probabilities=args.inception_probabilities,
            clap_model_dir=args.clap_model_dir,
            seed=args.seed, adapter_checkpoint=args.adapter_checkpoint,
            symbolic_policy_checkpoint=args.symbolic_policy_checkpoint,
            symbolic_max_new_tokens=args.symbolic_max_new_tokens,
            metric_inputs=json.loads(Path(args.metric_inputs_json).read_text()) if args.metric_inputs_json else None,
            reconstruction_reference_audio_path=args.reconstruction_reference_audio,
            inference_strategy=args.inference_strategy, inference_max_calls=args.inference_max_calls,
            inference_iterations=args.inference_iterations,
            local_audio_window_seconds=args.local_audio_window_seconds,
        )
    else:
        blueprint = load_blueprint_json(args.blueprint_json)
        run_pipeline_from_blueprint(
            blueprint=blueprint,
            output_dir=args.output_dir,
            refine_with_rl=args.refine_with_rl,
            refine_with_hierarchical_rl=args.refine_with_hierarchical_rl,
            rl_episodes=args.rl_episodes,
            enable_repair_cycle=args.enable_repair_cycle,
            critic_feedback_mode=args.critic_feedback_mode,
            human_feedback=human_feedback,
            llm_feedback=llm_feedback,
            input_mode="blueprint_json",
            generate_symbolic=args.generate_symbolic,
            generate_audio=args.generate_audio,
            mupt_model_dir=args.mupt_model_dir,
            mupt_device=args.mupt_device,
            musicgen_model_dir=args.musicgen_model_dir,
            musicgen_device=args.musicgen_device,
            rl_reward_mode=args.rl_reward_mode,
            symbolic_warmup_episodes=args.symbolic_warmup_episodes,
            update_symbolic_during_joint=args.update_symbolic_during_joint,
            llm_critic_base_url=args.llm_critic_base_url,
            llm_critic_api_key=args.llm_critic_api_key,
            llm_critic_model=args.llm_critic_model,
            user_prompt=args.user_prompt,
            reference_audio_path=args.reference_audio,
            evaluate_metrics=args.evaluate_paper_metrics,
            fad_reference_dir=args.fad_reference_dir,
            imagebind_video_embedding=args.imagebind_video_embedding,
            imagebind_audio_embedding=args.imagebind_audio_embedding,
            inception_probabilities=args.inception_probabilities,
            clap_model_dir=args.clap_model_dir,
            seed=args.seed, adapter_checkpoint=args.adapter_checkpoint,
            symbolic_policy_checkpoint=args.symbolic_policy_checkpoint,
            symbolic_max_new_tokens=args.symbolic_max_new_tokens,
            metric_inputs=json.loads(Path(args.metric_inputs_json).read_text()) if args.metric_inputs_json else None,
            reconstruction_reference_audio_path=args.reconstruction_reference_audio,
            inference_strategy=args.inference_strategy, inference_max_calls=args.inference_max_calls,
            inference_iterations=args.inference_iterations,
            local_audio_window_seconds=args.local_audio_window_seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
