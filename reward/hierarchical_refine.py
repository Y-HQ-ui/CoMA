from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import torch

from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_controls_to_prompts
from planning.blueprint_schema import MusicBlueprint
from planning.blueprint_to_abc import blueprint_to_bar_contexts, blueprint_to_mupt_prefixes
from .agent import RLAgent
from .config import RLConfig, build_default_rl_config
from .critic import build_critic_decision
from .hierarchical_env import HierarchicalCoMAEnv
from .repair import (
    execute_repair_cycle,
    repair_blueprint_deterministic,
    repair_render_controls_deterministic,
)
from .repair_trigger import RepairTriggerPolicy
from .rewards import MultiSourceRewardAggregator, SymbolicAudioConsistencyReward, SymbolicRewardCalculator
from .evidence import RewardEvidenceEvaluator
from .llm_critic_client import LLMStructuredCritic
from .llm_controller import LLMWeightController


def _select_rewards(reward_dict: Dict[str, float], *, phase: str) -> tuple[float, float]:
    if phase == "symbolic_warmup":
        return float(reward_dict["R_sym"]), 0.0
    shared_reward = float(reward_dict["R_total"])
    return shared_reward, shared_reward


def _build_reward_dict(
    reward_calc_sym: SymbolicRewardCalculator,
    reward_calc_cons: SymbolicAudioConsistencyReward,
    aggregator: MultiSourceRewardAggregator,
    info: Dict[str, Any],
) -> Dict[str, float]:
    symbolic_data = info.get("symbolic_data", {})
    consistency_data = info.get("consistency_data", {})
    visual_data = info.get("visual_data", {})
    musicality_data = info.get("musicality_data", {})

    r_sym = reward_calc_sym.compute_symbolic_reward(
        beat_data=symbolic_data.get("beat"),
        phrase_data=symbolic_data.get("phrase"),
        key_data=symbolic_data.get("key"),
        diversity_data=symbolic_data.get("diversity"),
        playability_data=symbolic_data.get("playability"),
    )
    r_cons = reward_calc_cons.compute_consistency_reward(
        onset_data=consistency_data.get("onset"),
        dynamic_data=consistency_data.get("dynamic"),
        color_data=consistency_data.get("color"),
    )
    r_vis = aggregator.compute_visual_rewards(
        motion_rhythm=visual_data.get("motion_rhythm", 0.0),
        color_tone=visual_data.get("color_tone", 0.0),
        novelty_thematic=visual_data.get("novelty_thematic", 0.0),
    )
    r_mus = aggregator.compute_musicality_rewards(
        tonal_stability=musicality_data.get("tonal_stability", 0.0),
        harmonic_progression=musicality_data.get("harmonic_progression", 0.0),
        rhythmic_density=musicality_data.get("rhythmic_density", 0.0),
        structural_symmetry=musicality_data.get("structural_symmetry", 0.0),
    )
    return {
        "R_sym": float(r_sym),
        "R_cons": float(r_cons),
        "R_vis": float(r_vis),
        "R_mus": float(r_mus),
        "R_total": float(aggregator.aggregate_rewards(r_vis, r_mus, r_cons)),
    }


def _shape_hierarchical_rewards(
    symbolic_reward: float,
    audio_reward: float,
    critic_decision: Any,
) -> tuple[float, float, Dict[str, float]]:
    deficits = critic_decision.diagnosis.get("deficits", {})
    scope = critic_decision.repair_plan.get("scope", "none")
    sym_bonus = 0.0
    aud_bonus = 0.0

    if scope == "blueprint":
        focus = float(deficits.get("visual", 0.0) + 0.5 * deficits.get("musicality", 0.0))
        sym_bonus = 0.12 * focus * max(symbolic_reward, 0.1)
        aud_bonus = 0.08 * focus * max(audio_reward, 0.1)
    elif scope == "symbolic":
        focus = float(deficits.get("symbolic", 0.0) + 0.5 * deficits.get("musicality", 0.0))
        sym_bonus = 0.18 * focus * max(symbolic_reward, 0.1)
        aud_bonus = -0.05 * focus * max(audio_reward, 0.1)
    elif scope == "audio":
        focus = float(deficits.get("consistency", 0.0) + 0.25 * deficits.get("visual", 0.0))
        sym_bonus = -0.03 * focus * max(symbolic_reward, 0.1)
        aud_bonus = 0.18 * focus * max(audio_reward, 0.1)

    shaped_symbolic = float(max(0.0, symbolic_reward + sym_bonus))
    shaped_audio = float(max(0.0, audio_reward + aud_bonus))
    return shaped_symbolic, shaped_audio, {
        "scope": scope,
        "symbolic_bonus": float(sym_bonus),
        "audio_bonus": float(aud_bonus),
    }


def _repair_trigger_score(
    trigger_policy: RepairTriggerPolicy,
    critic_decision: Any,
    *,
    symbolic_log_prob: float,
    audio_log_prob: float,
) -> Dict[str, float]:
    symbolic_confidence = float(np.clip(np.exp(symbolic_log_prob), 0.0, 1.0))
    audio_confidence = float(np.clip(np.exp(audio_log_prob), 0.0, 1.0))
    combined_confidence = 0.5 * (symbolic_confidence + audio_confidence)
    prediction = trigger_policy.predict(
        critic_decision,
        confidence=combined_confidence,
    )
    target = (
        critic_decision.diagnosis.get("repair_required", False)
        and prediction.heuristic_score >= 1.0
    )
    update_stats = trigger_policy.update(
        critic_decision,
        confidence=combined_confidence,
        target=bool(target),
    )
    return {
        "score": prediction.blended_score,
        "trigger_probability": prediction.trigger_prob,
        "heuristic_score": prediction.heuristic_score,
        "deficit_sum": prediction.deficit_sum,
        "symbolic_confidence": symbolic_confidence,
        "audio_confidence": audio_confidence,
        "uncertainty": prediction.uncertainty,
        "human_override": prediction.human_override,
        "policy_trigger": prediction.policy_trigger,
        "triggered": bool(
            critic_decision.diagnosis.get("repair_required", False)
            and prediction.policy_trigger
        ),
        "update_loss": update_stats["loss"],
    }


def _apply_online_repair(
    env: HierarchicalCoMAEnv,
    critic_decision: Any,
    *,
    max_passes: int,
) -> Dict[str, Any]:
    scope = critic_decision.repair_plan.get("scope", "none")
    num_passes = max(1, min(max_passes, int(critic_decision.repair_plan.get("num_passes", 1) or 1)))
    diagnosis = critic_decision.diagnosis

    if scope == "none":
        return {
            "applied": False,
            "scope": scope,
            "num_passes": 0,
        }

    repaired_blueprint = env.current_blueprint
    repaired_controls = [dict(control) for control in env.current_controls]

    if scope == "blueprint":
        repaired_blueprint = repair_blueprint_deterministic(
            repaired_blueprint,
            diagnosis,
            scope="blueprint",
            num_passes=num_passes,
        )
        repaired_controls = repair_render_controls_deterministic(
            blueprint_to_render_controls(repaired_blueprint),
            diagnosis,
            scope="symbolic",
            num_passes=num_passes,
        )
        repaired_controls = repair_render_controls_deterministic(
            repaired_controls,
            diagnosis,
            scope="audio",
            num_passes=num_passes,
        )
    elif scope == "symbolic":
        repaired_blueprint = repair_blueprint_deterministic(
            repaired_blueprint,
            diagnosis,
            scope="symbolic",
            num_passes=num_passes,
        )
        repaired_controls = repair_render_controls_deterministic(
            blueprint_to_render_controls(repaired_blueprint),
            diagnosis,
            scope="symbolic",
            num_passes=num_passes,
        )
    elif scope == "audio":
        repaired_controls = repair_render_controls_deterministic(
            repaired_controls,
            diagnosis,
            scope="audio",
            num_passes=num_passes,
        )

    env.current_blueprint = repaired_blueprint
    env.current_controls = repaired_controls
    return {
        "applied": True,
        "scope": scope,
        "num_passes": num_passes,
        "actions": list(critic_decision.repair_plan.get("actions", [])),
        "index_after_repair": env.index,
    }


def _hierarchical_reward_proxy(
    env: HierarchicalCoMAEnv,
    *,
    step_index: int,
    symbolic_control: Dict[str, Any],
    audio_control: Dict[str, Any],
    symbolic_action_name: str,
    audio_action_name: str,
) -> Dict[str, float]:
    symbolic_spec = next(
        (spec for spec in env.symbolic_actions if spec.name == symbolic_action_name),
        env.symbolic_actions[0],
    )
    audio_spec = next(
        (spec for spec in env.audio_actions if spec.name == audio_action_name),
        env.audio_actions[0],
    )
    target = env.base_controls[step_index]
    prior = target["video_prior_5d"]
    target_energy = 0.55 * prior["motion_intensity"] + 0.45 * prior["cut_density"]
    target_brightness = 0.55 * prior["brightness"] + 0.45 * prior["color_warmth"]
    target_tempo = int(80 + 70 * target_energy)
    motion_rhythm = 1.0 - min(1.0, abs(audio_control["tempo_bpm"] - target_tempo) / 60.0)
    color_tone = 1.0 - min(1.0, abs(audio_control.get("brightness_hint", prior["brightness"]) - target_brightness))
    novelty_thematic = 1.0 - min(1.0, abs(audio_control["energy"] - target_energy) + 0.5 * abs(audio_spec.texture_bias))
    symbolic_key = symbolic_control.get("key", symbolic_control.get("global_key", target["key"]))
    tonal_stability = 1.0 if symbolic_key == target["key"] else 0.5
    harmonic_progression = min(1.0, len(symbolic_control["chords"]) / 4.0)
    rhythmic_density = 1.0 - min(1.0, abs(symbolic_control["energy"] - target_energy))
    structural_symmetry = 1.0 - min(1.0, abs(symbolic_control["segment_id"] - target["segment_id"]) / max(1, len(env.current_controls)))
    symbolic_score = float(
        np.clip(
            0.40 * motion_rhythm
            + 0.15 * rhythmic_density
            + 0.15 * tonal_stability
            + 0.10 * harmonic_progression
            + 0.10 * (1.0 - min(1.0, abs(symbolic_control["energy"] - target_energy)))
            + 0.10 * (1.0 - min(1.0, abs(len(symbolic_control["chords"]) / 4.0 - 0.5))),
            0.0,
            1.0,
        )
    )
    audio_score = float(
        np.clip(
            0.40 * motion_rhythm
            + 0.25 * color_tone
            + 0.20 * novelty_thematic
            + 0.15 * (1.0 - min(1.0, abs(audio_control.get("texture_bias", 0.0)))),
            0.0,
            1.0,
        )
    )
    total = float(np.clip(0.5 * symbolic_score + 0.5 * audio_score, 0.0, 1.0))
    return {
        "R_total": total,
        "R_sym": symbolic_score,
        "R_cons": audio_score,
        "R_vis": motion_rhythm,
        "R_mus": 0.5 * (harmonic_progression + rhythmic_density),
    }


def refine_blueprint_hierarchical(
    blueprint: MusicBlueprint,
    *,
    config: RLConfig | None = None,
    num_episodes: int | None = None,
    device: str | None = None,
    human_feedback: Dict[str, Any] | None = None,
    llm_feedback: Dict[str, Any] | None = None,
    reward_evaluator: RewardEvidenceEvaluator | None = None,
) -> Dict[str, Any]:
    cfg = config or build_default_rl_config()
    env = HierarchicalCoMAEnv(
        blueprint,
        reward_evaluator=reward_evaluator,
        audio_window_seconds=cfg.audio_window_seconds,
    )
    llm_critic = LLMStructuredCritic.from_config(cfg)
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    episodes = num_episodes or cfg.num_episodes

    symbolic_agent = RLAgent(
        state_dim=env.symbolic_state_dim,
        action_dim=env.symbolic_action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        device=run_device,
        hidden_dim=cfg.hidden_dim,
    )
    audio_agent = RLAgent(
        state_dim=env.audio_state_dim,
        action_dim=env.audio_action_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        device=run_device,
        hidden_dim=cfg.hidden_dim,
    )

    symbolic_reward_calc = SymbolicRewardCalculator(cfg)
    consistency_reward_calc = SymbolicAudioConsistencyReward(cfg)
    reward_aggregator = MultiSourceRewardAggregator(cfg)
    trigger_policy = RepairTriggerPolicy(cfg, device=run_device)
    weight_controller = LLMWeightController(
        cfg,
        llm_critic.suggest_weights if llm_critic is not None else None,
    )

    history: Dict[str, List[Any]] = {
        "symbolic_episode_rewards": [],
        "audio_episode_rewards": [],
        "combined_episode_rewards": [],
        "reward_components": [],
        "step_critic_decisions": [],
        "step_reward_shaping": [],
        "online_repair_events": [],
        "repair_trigger_updates": [],
        "symbolic_ppo_stats": [],
        "audio_ppo_stats": [],
        "training_phase": [],
        "weight_history": [],
        "feedback_history": [],
        "gradient_free_exploration": [],
        "consistency_weight_schedule": [],
    }

    for episode_idx in range(episodes):
        phase = "symbolic_warmup" if episode_idx < min(cfg.symbolic_warmup_episodes, episodes) else "joint"
        history["training_phase"].append(phase)
        exploration_scale = cfg.gradient_free_logit_noise * (cfg.gradient_free_noise_decay ** episode_idx)
        if phase == "symbolic_warmup":
            consistency_scale = 0.0
        else:
            warmup = min(cfg.symbolic_warmup_episodes, episodes)
            joint_progress = (episode_idx - warmup) / max(1, episodes - warmup - 1)
            consistency_scale = 1.0 + cfg.consistency_weight_ramp * float(np.clip(joint_progress, 0.0, 1.0))
        current_weights = weight_controller.get_current_weights()
        reward_aggregator.update_weights(
            current_weights["alpha_vis"],
            current_weights["beta_mus"],
            float(current_weights["gamma_cons"]) * consistency_scale,
        )
        history["gradient_free_exploration"].append(
            {"episode": episode_idx, "logit_noise_scale": exploration_scale}
        )
        history["consistency_weight_schedule"].append(
            {"episode": episode_idx, "phase": phase, "scale": consistency_scale}
        )
        symbolic_state, audio_state = env.reset()
        done = False
        repairs_used = 0

        symbolic_states = []
        symbolic_actions = []
        symbolic_log_probs = []
        symbolic_values = []
        symbolic_rewards = []
        symbolic_dones = []

        audio_states = []
        audio_actions = []
        audio_log_probs = []
        audio_values = []
        audio_rewards = []
        audio_dones = []

        episode_components: List[Dict[str, float]] = []
        episode_step_decisions: List[Dict[str, Any]] = []
        episode_shaping: List[Dict[str, float]] = []
        episode_repairs: List[Dict[str, Any]] = []

        while not done:
            symbolic_action, symbolic_log_prob, symbolic_value = symbolic_agent.select_action(
                symbolic_state,
                logit_noise_scale=exploration_scale,
            )
            audio_window_states: List[np.ndarray] = []
            audio_window_actions: List[int] = []
            audio_window_log_probs: List[float] = []
            audio_window_values: List[float] = []
            window_count = max(1, len(env.current_controls[env.index].get("render_windows", [])))
            if phase == "symbolic_warmup":
                audio_action: int | List[int] = [0] * window_count
                audio_log_prob, audio_value = 0.0, 0.0
            else:
                for window_index in range(window_count):
                    window_state = env.get_audio_state(window_index)
                    window_action, window_log_prob, window_value = audio_agent.select_action(
                        window_state,
                        logit_noise_scale=exploration_scale,
                    )
                    audio_window_states.append(window_state)
                    audio_window_actions.append(window_action)
                    audio_window_log_probs.append(window_log_prob)
                    audio_window_values.append(window_value)
                audio_action = audio_window_actions
                audio_log_prob = float(np.mean(audio_window_log_probs))
                audio_value = float(np.mean(audio_window_values))

            next_symbolic_state, next_audio_state, _, _, done, info = env.step(symbolic_action, audio_action)
            reward_dict = _build_reward_dict(symbolic_reward_calc, consistency_reward_calc, reward_aggregator, info)
            global_step = episode_idx * max(1, len(env.current_controls)) + len(episode_components)
            if global_step % cfg.llm_update_freq == 0:
                weights = weight_controller.update_weights(
                    R_vis=reward_dict["R_vis"],
                    R_mus=reward_dict["R_mus"],
                    R_cons=reward_dict["R_cons"],
                    H_human=human_feedback,
                    H_llm=llm_feedback,
                    step=global_step,
                )
                reward_aggregator.update_weights(weights[0], weights[1], weights[2] * consistency_scale)
                history["weight_history"].append(
                    {"step": global_step,
                     "weights": {"alpha_vis": reward_aggregator.alpha_vis.copy(),
                                 "beta_mus": reward_aggregator.beta_mus.copy(),
                                 "gamma_cons": reward_aggregator.gamma_cons},
                     "proposed_weights": weight_controller.get_current_weights(),
                     "normalization": "single_global_simplex_after_consistency_ramp"}
                )
                if weight_controller.feedback_history:
                    history["feedback_history"].append(weight_controller.feedback_history[-1])
                reward_dict = _build_reward_dict(symbolic_reward_calc, consistency_reward_calc, reward_aggregator, info)
            symbolic_reward, audio_reward = _select_rewards(reward_dict, phase=phase)
            step_critic_decision = build_critic_decision(
                reward_dict,
                feedback_mode=cfg.critic_feedback_mode,
                global_threshold=cfg.repair_global_threshold,
                component_threshold=cfg.repair_component_threshold,
                max_repair_passes=cfg.max_repair_passes,
                human_feedback=human_feedback,
                llm_feedback=llm_feedback,
                llm_critic=llm_critic,
                critic_context={
                    "episode": episode_idx,
                    "phase": phase,
                    "symbolic_action": info.get("symbolic_action_name"),
                    "audio_action": info.get("audio_action_names", info.get("audio_action_name")),
                    "evidence_source": info.get("evidence_source"),
                },
            )
            symbolic_reward, audio_reward, shaping_info = _shape_hierarchical_rewards(
                symbolic_reward,
                audio_reward,
                step_critic_decision,
            )
            trigger_info = _repair_trigger_score(
                trigger_policy,
                step_critic_decision,
                symbolic_log_prob=symbolic_log_prob,
                audio_log_prob=audio_log_prob,
            )
            step_index = max(0, env.index - 1)
            pre_symbolic_control = dict(env.current_blueprint.segments[step_index].to_dict())
            pre_audio_control = dict(env.current_controls[step_index]) if step_index < len(env.current_controls) else {}

            symbolic_states.append(symbolic_state)
            symbolic_actions.append(symbolic_action)
            symbolic_log_probs.append(symbolic_log_prob)
            symbolic_values.append(symbolic_value)
            symbolic_rewards.append(symbolic_reward)
            symbolic_dones.append(done)

            if phase == "joint":
                per_window_reward = audio_reward / max(1, window_count)
                for window_index in range(window_count):
                    audio_states.append(audio_window_states[window_index])
                    audio_actions.append(audio_window_actions[window_index])
                    audio_log_probs.append(audio_window_log_probs[window_index])
                    audio_values.append(audio_window_values[window_index])
                    audio_rewards.append(per_window_reward)
                    audio_dones.append(bool(done and window_index == window_count - 1))

            episode_components.append(reward_dict)
            episode_step_decisions.append(
                {
                    "episode": episode_idx,
                    "step": len(episode_components) - 1,
                    "feedback_source": step_critic_decision.feedback_source,
                    "diagnosis": step_critic_decision.diagnosis,
                    "repair_plan": step_critic_decision.repair_plan,
                }
            )
            episode_shaping.append(shaping_info)
            episode_shaping[-1].update(
                {
                    "repair_trigger_score": trigger_info["score"],
                    "trigger_probability": trigger_info["trigger_probability"],
                    "heuristic_trigger_score": trigger_info["heuristic_score"],
                    "symbolic_confidence": trigger_info["symbolic_confidence"],
                    "audio_confidence": trigger_info["audio_confidence"],
                    "uncertainty": trigger_info["uncertainty"],
                    "human_override": trigger_info["human_override"],
                    "trigger_update_loss": trigger_info["update_loss"],
                    "outcome_bonus": 0.0,
                    "outcome_update_loss": 0.0,
                }
            )

            if (
                step_critic_decision.diagnosis.get("repair_required", False)
                and trigger_info["triggered"]
                and repairs_used < cfg.max_repair_passes
            ):
                repair_event = _apply_online_repair(
                    env,
                    step_critic_decision,
                    max_passes=1,
                )
                post_symbolic_control = dict(env.current_blueprint.segments[step_index].to_dict()) if step_index < len(env.current_blueprint.segments) else pre_symbolic_control
                post_audio_control = dict(env.current_controls[step_index]) if step_index < len(env.current_controls) else pre_audio_control
                after_evidence = env.reward_evaluator.evaluate(
                    env.current_blueprint,
                    step_index,
                    post_audio_control,
                )
                after_reward = _build_reward_dict(
                    symbolic_reward_calc,
                    consistency_reward_calc,
                    reward_aggregator,
                    after_evidence,
                )
                outcome_delta = float(after_reward["R_total"] - reward_dict["R_total"])
                outcome_bonus = 0.25 if repair_event.get("applied", False) else -0.05
                outcome_bonus = float(np.clip(outcome_delta, -0.25, 0.5))
                outcome_stats = trigger_policy.update(
                    step_critic_decision,
                    confidence=0.5 * (trigger_info["symbolic_confidence"] + trigger_info["audio_confidence"]),
                    target=bool(repair_event.get("applied", False)),
                    outcome_bonus=outcome_bonus,
                )
                repair_event["episode"] = episode_idx
                repair_event["step"] = len(episode_components) - 1
                repair_event["feedback_source"] = step_critic_decision.feedback_source
                repair_event["trigger_score"] = trigger_info["score"]
                repair_event["trigger_probability"] = trigger_info["trigger_probability"]
                repair_event["heuristic_trigger_score"] = trigger_info["heuristic_score"]
                repair_event["symbolic_confidence"] = trigger_info["symbolic_confidence"]
                repair_event["audio_confidence"] = trigger_info["audio_confidence"]
                repair_event["uncertainty"] = trigger_info["uncertainty"]
                repair_event["human_override"] = trigger_info["human_override"]
                repair_event["trigger_update_loss"] = trigger_info["update_loss"]
                repair_event["outcome_bonus"] = outcome_bonus
                repair_event["outcome_delta"] = outcome_delta
                repair_event["outcome_update_loss"] = outcome_stats["loss"]
                episode_shaping[-1]["outcome_bonus"] = outcome_bonus
                episode_shaping[-1]["outcome_delta"] = outcome_delta
                episode_shaping[-1]["outcome_update_loss"] = outcome_stats["loss"]
                episode_repairs.append(repair_event)
                if repair_event["applied"]:
                    repairs_used += 1
                    if not done:
                        next_symbolic_state = env.get_symbolic_state()
                        next_audio_state = env.get_audio_state()

            symbolic_state = next_symbolic_state
            audio_state = next_audio_state

        sym_stats: Dict[str, float] = {}
        if phase == "symbolic_warmup" or cfg.update_symbolic_during_joint:
            symbolic_batch = symbolic_agent.build_rollout_batch(
                symbolic_states,
                symbolic_actions,
                symbolic_log_probs,
                symbolic_rewards,
                symbolic_values,
                symbolic_dones,
                last_value=0.0,
                gamma=cfg.symbolic_gamma,
                gae_lambda=cfg.gae_lambda,
            )
            sym_stats = symbolic_agent.update_policy(
                symbolic_batch,
                ppo_clip=cfg.ppo_clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
                max_grad_norm=cfg.max_grad_norm,
                update_epochs=cfg.update_epochs,
                minibatch_size=min(cfg.batch_size, len(symbolic_states)),
            )
        aud_stats: Dict[str, float] = {}
        if phase == "joint" and audio_states:
            audio_batch = audio_agent.build_rollout_batch(
                audio_states,
                audio_actions,
                audio_log_probs,
                audio_rewards,
                audio_values,
                audio_dones,
                last_value=0.0,
                gamma=cfg.audio_gamma,
                gae_lambda=cfg.gae_lambda,
            )
            aud_stats = audio_agent.update_policy(
                audio_batch,
                ppo_clip=cfg.ppo_clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
                max_grad_norm=cfg.max_grad_norm,
                update_epochs=cfg.update_epochs,
                minibatch_size=min(cfg.batch_size, len(audio_states)),
            )

        sym_total = float(sum(symbolic_rewards))
        aud_total = float(sum(audio_rewards))
        history["symbolic_episode_rewards"].append(sym_total)
        history["audio_episode_rewards"].append(aud_total)
        history["combined_episode_rewards"].append(sym_total if phase == "symbolic_warmup" else 0.5 * (sym_total + aud_total))
        history["reward_components"].append(episode_components[-1] if episode_components else {})
        history["step_critic_decisions"].append(episode_step_decisions)
        history["step_reward_shaping"].append(episode_shaping)
        history["online_repair_events"].append(episode_repairs)
        history["repair_trigger_updates"].append(list(trigger_policy.training_history[-len(episode_components):]) if episode_components else [])
        history["symbolic_ppo_stats"].append(sym_stats)
        history["audio_ppo_stats"].append(aud_stats)

    refined_blueprint = env.current_blueprint
    refined_controls = env.current_controls
    base_controls = blueprint_to_render_controls(blueprint)
    base_prefixes = blueprint_to_mupt_prefixes(blueprint)
    refined_prefixes = blueprint_to_mupt_prefixes(refined_blueprint)
    refined_bar_contexts = blueprint_to_bar_contexts(refined_blueprint)
    final_reward = history["reward_components"][-1] if history["reward_components"] else {
        "R_total": 0.0,
        "R_vis": 0.0,
        "R_mus": 0.0,
        "R_cons": 0.0,
        "R_sym": 0.0,
    }
    critic_decision = build_critic_decision(
        final_reward,
        feedback_mode=cfg.critic_feedback_mode,
        global_threshold=cfg.repair_global_threshold,
        component_threshold=cfg.repair_component_threshold,
        max_repair_passes=cfg.max_repair_passes,
        human_feedback=human_feedback,
        llm_feedback=llm_feedback,
        llm_critic=llm_critic,
        critic_context={"stage": "final_hierarchical", "num_segments": len(refined_blueprint.segments)},
    )
    repair_output = execute_repair_cycle(
        refined_blueprint,
        final_reward,
        feedback_mode=cfg.critic_feedback_mode,
        global_threshold=cfg.repair_global_threshold,
        component_threshold=cfg.repair_component_threshold,
        max_repair_passes=cfg.max_repair_passes,
        human_feedback=human_feedback,
        llm_feedback=llm_feedback,
        llm_critic=llm_critic,
        critic_context={"stage": "repair_hierarchical", "num_segments": len(refined_blueprint.segments)},
    )

    return {
        "config": __import__("reward.config", fromlist=["public_config"]).public_config(cfg),
        "optimization_target": "control_MLPs_not_MuPT_or_MusicGen_parameters",
        "num_episodes": episodes,
        "refined_blueprint": refined_blueprint.to_dict(),
        "base_controls": base_controls,
        "refined_controls": refined_controls,
        "base_mupt_prefixes": base_prefixes,
        "refined_mupt_prefixes": refined_prefixes,
        "refined_bar_contexts": refined_bar_contexts,
        "base_prompts": render_controls_to_prompts(base_controls),
        "refined_prompts": render_controls_to_prompts(refined_controls),
        "symbolic_action_history": list(env.symbolic_action_history),
        "audio_action_history": list(env.audio_action_history),
        "training_history": history,
        "symbolic_agent_stats": symbolic_agent.training_stats,
        "audio_agent_stats": audio_agent.training_stats,
        "critic_decision": {
            "feedback_source": critic_decision.feedback_source,
            "diagnosis": critic_decision.diagnosis,
            "repair_plan": critic_decision.repair_plan,
        },
        "repair_output": repair_output,
    }
