from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List
import torch
from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_controls_to_prompts
from planning.blueprint_schema import MusicBlueprint
from .agent import RLAgent
from .coma_env import CoMARLEnv
from .config import RLConfig, build_default_rl_config
from .critic import build_critic_decision
from .hierarchical_refine import refine_blueprint_hierarchical
from .repair import execute_repair_cycle
from .trainer import RLTrainer
from .evidence import RewardEvidenceEvaluator
from .llm_critic_client import LLMStructuredCritic

def refine_blueprint_controls(blueprint: MusicBlueprint, *, config: RLConfig | None=None, num_episodes: int | None=None, device: str | None=None, human_feedback: Dict[str, Any] | None=None, llm_feedback: Dict[str, Any] | None=None, reward_evaluator: RewardEvidenceEvaluator | None=None) -> Dict[str, Any]:
    cfg = config or build_default_rl_config()
    env = CoMARLEnv(blueprint, reward_evaluator=reward_evaluator)
    llm_critic = LLMStructuredCritic.from_config(cfg)
    run_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    episodes = num_episodes or cfg.num_episodes
    agent = RLAgent(state_dim=env.state_dim, action_dim=env.action_dim, learning_rate=cfg.learning_rate, gamma=cfg.gamma, device=run_device, hidden_dim=cfg.hidden_dim)
    trainer = RLTrainer(config=cfg, agent=agent, env_interface=env, llm_interface=llm_critic.suggest_weights if llm_critic is not None else None, llm_critic=llm_critic, default_human_feedback=human_feedback, default_llm_feedback=llm_feedback)
    trainer.train(num_episodes=episodes)
    refined_controls = [dict(control) for control in env.current_controls]
    base_controls = blueprint_to_render_controls(blueprint)
    final_reward = trainer.training_history['reward_components'][-1] if trainer.training_history['reward_components'] else {'R_total': 0.0, 'R_vis': 0.0, 'R_mus': 0.0, 'R_cons': 0.0, 'R_sym': 0.0}
    critic_decision = build_critic_decision(final_reward, feedback_mode=cfg.critic_feedback_mode, global_threshold=cfg.repair_global_threshold, component_threshold=cfg.repair_component_threshold, max_repair_passes=cfg.max_repair_passes, human_feedback=human_feedback, llm_feedback=llm_feedback, llm_critic=llm_critic, critic_context={'stage': 'final_single_level', 'num_segments': len(blueprint.segments)})
    repair_output = execute_repair_cycle(blueprint, final_reward, feedback_mode=cfg.critic_feedback_mode, global_threshold=cfg.repair_global_threshold, component_threshold=cfg.repair_component_threshold, max_repair_passes=cfg.max_repair_passes, human_feedback=human_feedback, llm_feedback=llm_feedback, llm_critic=llm_critic, critic_context={'stage': 'repair_single_level', 'num_segments': len(blueprint.segments)})
    return {'config': __import__('reward.config', fromlist=['public_config']).public_config(cfg), 'num_episodes': episodes, 'base_controls': base_controls, 'refined_controls': refined_controls, 'base_prompts': render_controls_to_prompts(base_controls), 'refined_prompts': render_controls_to_prompts(refined_controls), 'training_history': trainer.training_history, 'agent_stats': agent.training_stats, 'critic_decision': {'feedback_source': critic_decision.feedback_source, 'diagnosis': critic_decision.diagnosis, 'repair_plan': critic_decision.repair_plan}, 'repair_output': repair_output}
__all__ = ['refine_blueprint_controls', 'refine_blueprint_hierarchical']
