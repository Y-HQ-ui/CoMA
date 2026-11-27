import numpy as np
from typing import Dict, List, Optional, Any, Callable
import torch
try:
    from .config import RLConfig
    from .rewards import SymbolicRewardCalculator, SymbolicAudioConsistencyReward, MultiSourceRewardAggregator
    from .critic import build_critic_decision
    from .llm_controller import LLMWeightController
    from .repair import repair_render_controls_deterministic
    from .repair_trigger import RepairTriggerPolicy
except ImportError:
    from config import RLConfig
    from rewards import SymbolicRewardCalculator, SymbolicAudioConsistencyReward, MultiSourceRewardAggregator
    from critic import build_critic_decision
    from llm_controller import LLMWeightController
    from repair import repair_render_controls_deterministic
    from repair_trigger import RepairTriggerPolicy

class RLTrainer:

    def __init__(self, config: RLConfig, agent: Any, env_interface: Optional[Callable]=None, llm_interface: Optional[Callable]=None, llm_critic: Any=None, default_human_feedback: Optional[Dict[str, Any]]=None, default_llm_feedback: Optional[Dict[str, Any]]=None):
        self.config = config
        self.agent = agent
        self.env_interface = env_interface
        self.default_human_feedback = default_human_feedback
        self.default_llm_feedback = default_llm_feedback
        self.llm_critic = llm_critic
        self.symbolic_reward_calc = SymbolicRewardCalculator(config)
        self.consistency_reward_calc = SymbolicAudioConsistencyReward(config)
        self.reward_aggregator = MultiSourceRewardAggregator(config)
        self.llm_controller = LLMWeightController(config, llm_interface)
        self.repair_trigger_policy = RepairTriggerPolicy(config, device=str(self.agent.device))
        self.training_history = {'episode_rewards': [], 'episode_lengths': [], 'reward_components': [], 'weight_history': [], 'critic_decisions': [], 'step_reward_shaping': [], 'online_repair_events': [], 'feedback_history': [], 'ppo_stats': []}

    def _repair_trigger_score(self, critic_decision: Any, *, log_prob: float) -> Dict[str, float]:
        action_confidence = float(np.clip(np.exp(log_prob), 0.0, 1.0))
        prediction = self.repair_trigger_policy.predict(critic_decision, confidence=action_confidence)
        target = critic_decision.diagnosis.get('repair_required', False) and prediction.heuristic_score >= 1.0
        update_stats = self.repair_trigger_policy.update(critic_decision, confidence=action_confidence, target=bool(target))
        return {'score': prediction.blended_score, 'trigger_probability': prediction.trigger_prob, 'heuristic_score': prediction.heuristic_score, 'deficit_sum': prediction.deficit_sum, 'action_confidence': prediction.action_confidence, 'uncertainty': prediction.uncertainty, 'human_override': prediction.human_override, 'policy_trigger': prediction.policy_trigger, 'triggered': bool(critic_decision.diagnosis.get('repair_required', False) and prediction.policy_trigger), 'update_loss': update_stats['loss']}

    def _single_level_reward_proxy(self, *, step_index: int, control: Dict[str, Any], action_name: str) -> Dict[str, float]:
        if not self.env_interface or not hasattr(self.env_interface, 'base_controls'):
            return {'R_total': 0.0}
        target = self.env_interface.base_controls[step_index]
        prior = target['video_prior_5d']
        action_spec = next((spec for spec in getattr(self.env_interface, 'action_specs', []) if spec.name == action_name), getattr(self.env_interface, 'action_specs', [None])[0])
        action_texture = getattr(action_spec, 'texture_bias', 0.0) if action_spec is not None else 0.0
        target_energy = 0.55 * prior['motion_intensity'] + 0.45 * prior['cut_density']
        target_brightness = 0.55 * prior['brightness'] + 0.45 * prior['color_warmth']
        target_tempo = int(80 + 70 * target_energy)
        motion_rhythm = 1.0 - min(1.0, abs(control['tempo_bpm'] - target_tempo) / 60.0)
        color_tone = 1.0 - min(1.0, abs(control.get('brightness_hint', prior['brightness']) - target_brightness))
        novelty_thematic = 1.0 - min(1.0, abs(control['energy'] - target_energy) + 0.5 * abs(action_texture))
        tonal_stability = 1.0 if control['key'] == target['key'] else 0.5
        harmonic_progression = min(1.0, len(control['chords']) / 4.0)
        rhythmic_density = 1.0 - min(1.0, abs(control['energy'] - target_energy))
        structural_symmetry = 1.0 - min(1.0, abs(control['segment_id'] - target['segment_id']) / max(1, len(self.env_interface.current_controls)))
        total = 0.35 * motion_rhythm + 0.25 * color_tone + 0.2 * novelty_thematic
        total += 0.1 * tonal_stability + 0.1 * min(1.0, abs(control['energy'] - target['energy']) * -1.0 + 1.0)
        total = float(max(0.0, min(1.0, total)))
        return {'R_total': total, 'R_vis': float(0.35 * motion_rhythm + 0.25 * color_tone + 0.2 * novelty_thematic), 'R_mus': float(0.1 * tonal_stability + 0.1 * rhythmic_density), 'R_cons': float(0.1 * tonal_stability + 0.1 * rhythmic_density), 'R_sym': float(0.25 * motion_rhythm + 0.25 * rhythmic_density + 0.25 * tonal_stability + 0.25 * harmonic_progression)}

    def _shape_single_level_reward(self, reward: float, reward_dict: Dict[str, float], critic_decision: Any) -> tuple[float, Dict[str, float]]:
        deficits = critic_decision.diagnosis.get('deficits', {})
        scope = critic_decision.repair_plan.get('scope', 'none')
        bonus = 0.0
        if scope == 'blueprint':
            focus = float(deficits.get('visual', 0.0) + 0.5 * deficits.get('musicality', 0.0))
            bonus = 0.14 * focus * max(reward, 0.1)
        elif scope == 'symbolic':
            focus = float(deficits.get('symbolic', 0.0) + 0.5 * deficits.get('musicality', 0.0))
            bonus = 0.08 * focus * max(reward_dict.get('R_mus', reward), 0.1)
        elif scope == 'audio':
            focus = float(deficits.get('consistency', 0.0) + 0.25 * deficits.get('visual', 0.0))
            bonus = 0.18 * focus * max(reward_dict.get('R_cons', reward), 0.1)
        shaped_reward = float(max(0.0, reward + bonus))
        return (shaped_reward, {'scope': scope, 'bonus': float(bonus), 'base_reward': float(reward), 'shaped_reward': shaped_reward})

    def _apply_single_level_online_repair(self, critic_decision: Any) -> Dict[str, Any]:
        if not self.env_interface or not hasattr(self.env_interface, 'current_controls'):
            return {'applied': False, 'scope': critic_decision.repair_plan.get('scope', 'none'), 'reason': 'env_interface_has_no_current_controls'}
        scope = critic_decision.repair_plan.get('scope', 'none')
        if scope == 'none':
            return {'applied': False, 'scope': scope, 'reason': 'no_repair_required'}
        repair_scope = 'audio' if scope == 'audio' else 'symbolic'
        self.env_interface.current_controls = repair_render_controls_deterministic(self.env_interface.current_controls, critic_decision.diagnosis, scope=repair_scope, num_passes=1)
        return {'applied': True, 'scope': scope, 'repair_scope': repair_scope, 'actions': list(critic_decision.repair_plan.get('actions', [])), 'index_after_repair': getattr(self.env_interface, 'index', None)}

    def compute_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, float]:
        symbolic_data = info.get('symbolic_data', {})
        R_sym = self.symbolic_reward_calc.compute_symbolic_reward(beat_data=symbolic_data.get('beat'), phrase_data=symbolic_data.get('phrase'), key_data=symbolic_data.get('key'), diversity_data=symbolic_data.get('diversity'), playability_data=symbolic_data.get('playability'))
        consistency_data = info.get('consistency_data', {})
        R_sym_aud = self.consistency_reward_calc.compute_consistency_reward(onset_data=consistency_data.get('onset'), dynamic_data=consistency_data.get('dynamic'), color_data=consistency_data.get('color'))
        visual_data = info.get('visual_data', {})
        R_vis = self.reward_aggregator.compute_visual_rewards(motion_rhythm=visual_data.get('motion_rhythm', 0.0), color_tone=visual_data.get('color_tone', 0.0), novelty_thematic=visual_data.get('novelty_thematic', 0.0))
        musicality_data = info.get('musicality_data', {})
        R_mus = self.reward_aggregator.compute_musicality_rewards(tonal_stability=musicality_data.get('tonal_stability', 0.0), harmonic_progression=musicality_data.get('harmonic_progression', 0.0), rhythmic_density=musicality_data.get('rhythmic_density', 0.0), structural_symmetry=musicality_data.get('structural_symmetry', 0.0))
        kl_data = info.get('kl_data', {})
        if kl_data:
            R_cons_kl = self.reward_aggregator.compute_consistency_reward_kl(kl_data.get('p_y', np.array([0.5])), kl_data.get('q_y', np.array([0.5])))
        else:
            R_cons_kl = R_sym_aud
        R_total = self.reward_aggregator.aggregate_rewards(R_vis, R_mus, R_cons_kl)
        return {'R_sym': R_sym, 'R_sym_aud': R_sym_aud, 'R_vis': R_vis, 'R_mus': R_mus, 'R_cons': R_cons_kl, 'R_total': R_total}

    def train_episode(self, episode: int) -> Dict[str, Any]:
        state = self._get_initial_state()
        episode_rewards = []
        rollout_states = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_dones = []
        done = False
        step = 0
        repairs_used = 0
        reward_dict = {'R_total': 0.0, 'R_vis': 0.0, 'R_mus': 0.0, 'R_cons': 0.0}
        episode_step_shaping = []
        episode_online_repairs = []
        while not done and step < self.config.max_steps_per_episode:
            action, log_prob, value = self.agent.select_action(state)
            if self.env_interface:
                env_step = self.env_interface.step(action)
                if len(env_step) == 4:
                    next_state, env_reward, done, info = env_step
                else:
                    next_state, done, info = env_step
                    env_reward = None
            else:
                next_state, done, info = self._default_env_step(action, state)
                env_reward = None
            reward_dict = self.compute_reward(state, action, next_state, info)
            reward = float(reward_dict['R_total'])
            critic_decision = build_critic_decision(reward_dict, feedback_mode=self.config.critic_feedback_mode, global_threshold=self.config.repair_global_threshold, component_threshold=self.config.repair_component_threshold, max_repair_passes=self.config.max_repair_passes, human_feedback=info.get('human_feedback', self.default_human_feedback), llm_feedback=info.get('llm_feedback', self.default_llm_feedback), llm_critic=self.llm_critic, critic_context={'step': episode * self.config.max_steps_per_episode + step, 'action_name': info.get('action_name'), 'render_control': info.get('render_control'), 'evidence_source': info.get('evidence_source')})
            reward, shaping_info = self._shape_single_level_reward(reward, reward_dict, critic_decision)
            trigger_info = self._repair_trigger_score(critic_decision, log_prob=log_prob)
            step_index = max(0, getattr(self.env_interface, 'index', 1) - 1) if self.env_interface else 0
            pre_repair_control = None
            if self.env_interface and hasattr(self.env_interface, 'current_controls') and (step_index < len(self.env_interface.current_controls)):
                pre_repair_control = dict(self.env_interface.current_controls[step_index])
            episode_step_shaping.append({'step': episode * self.config.max_steps_per_episode + step, **shaping_info, 'repair_trigger_score': trigger_info['score'], 'trigger_probability': trigger_info['trigger_probability'], 'heuristic_trigger_score': trigger_info['heuristic_score'], 'action_confidence': trigger_info['action_confidence'], 'uncertainty': trigger_info['uncertainty'], 'human_override': trigger_info['human_override'], 'trigger_update_loss': trigger_info['update_loss'], 'outcome_bonus': 0.0, 'outcome_update_loss': 0.0})
            rollout_states.append(state)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value)
            rollout_dones.append(done)
            episode_rewards.append(reward)
            if step % self.config.llm_update_freq == 0:
                weights = self.llm_controller.update_weights(R_vis=reward_dict['R_vis'], R_mus=reward_dict['R_mus'], R_cons=reward_dict['R_cons'], H_human=info.get('human_feedback', self.default_human_feedback), H_llm=info.get('llm_feedback', self.default_llm_feedback), step=episode * self.config.max_steps_per_episode + step)
                controller_feedback = self.llm_controller.feedback_history[-1] if self.llm_controller.feedback_history else {}
                self.reward_aggregator.update_weights(weights[0], weights[1], weights[2])
                self.training_history['weight_history'].append({'step': episode * self.config.max_steps_per_episode + step, 'weights': self.llm_controller.get_current_weights(), 'feedback_source': controller_feedback.get('feedback_source', critic_decision.feedback_source)})
                self.training_history['feedback_history'].append(controller_feedback)
            self.training_history['critic_decisions'].append({'step': episode * self.config.max_steps_per_episode + step, 'feedback_source': critic_decision.feedback_source, 'diagnosis': critic_decision.diagnosis, 'repair_plan': critic_decision.repair_plan})
            if critic_decision.diagnosis.get('repair_required', False) and trigger_info['triggered'] and (repairs_used < self.config.max_repair_passes):
                repair_event = self._apply_single_level_online_repair(critic_decision)
                post_repair_control = None
                if self.env_interface and hasattr(self.env_interface, 'current_controls') and (step_index < len(self.env_interface.current_controls)):
                    post_repair_control = dict(self.env_interface.current_controls[step_index])
                before_proxy = self._single_level_reward_proxy(step_index=step_index, control=pre_repair_control or post_repair_control or {}, action_name=info.get('action_name', 'keep')) if pre_repair_control or post_repair_control else {'R_total': 0.0}
                after_proxy = self._single_level_reward_proxy(step_index=step_index, control=post_repair_control or pre_repair_control or {}, action_name=info.get('action_name', 'keep')) if pre_repair_control or post_repair_control else {'R_total': 0.0}
                outcome_delta = float(after_proxy['R_total'] - before_proxy['R_total'])
                outcome_bonus = float(np.clip(outcome_delta, -0.25, 0.5))
                outcome_stats = self.repair_trigger_policy.update(critic_decision, confidence=trigger_info['action_confidence'], target=bool(repair_event.get('applied', False)), outcome_bonus=outcome_bonus)
                repair_event.update({'step': episode * self.config.max_steps_per_episode + step, 'feedback_source': critic_decision.feedback_source, 'trigger_score': trigger_info['score'], 'trigger_probability': trigger_info['trigger_probability'], 'heuristic_trigger_score': trigger_info['heuristic_score'], 'action_confidence': trigger_info['action_confidence'], 'uncertainty': trigger_info['uncertainty'], 'human_override': trigger_info['human_override'], 'trigger_update_loss': trigger_info['update_loss'], 'outcome_bonus': outcome_bonus, 'outcome_delta': outcome_delta, 'outcome_update_loss': outcome_stats['loss']})
                episode_step_shaping[-1]['outcome_bonus'] = outcome_bonus
                episode_step_shaping[-1]['outcome_delta'] = outcome_delta
                episode_step_shaping[-1]['outcome_update_loss'] = outcome_stats['loss']
                episode_online_repairs.append(repair_event)
                if repair_event.get('applied', False):
                    repairs_used += 1
                    if not done and self.env_interface and hasattr(self.env_interface, '_state_from_index'):
                        next_state = self.env_interface._state_from_index(self.env_interface.index)
            state = next_state
            step += 1
        if done:
            last_value = 0.0
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
                last_value = float(self.agent.value_net(state_tensor).item())
        batch = self.agent.build_rollout_batch(rollout_states, rollout_actions, rollout_log_probs, episode_rewards, rollout_values, rollout_dones, last_value, self.config.gamma, self.config.gae_lambda)
        ppo_stats = self.agent.update_policy(batch, ppo_clip=self.config.ppo_clip, value_coef=self.config.value_coef, entropy_coef=self.config.entropy_coef, max_grad_norm=self.config.max_grad_norm, update_epochs=self.config.update_epochs, minibatch_size=self.config.batch_size)
        episode_total_reward = float(sum(episode_rewards))
        self.training_history['episode_rewards'].append(episode_total_reward)
        self.training_history['episode_lengths'].append(step)
        self.training_history['reward_components'].append(reward_dict)
        self.training_history['step_reward_shaping'].append(episode_step_shaping)
        self.training_history['online_repair_events'].append(episode_online_repairs)
        self.training_history['ppo_stats'].append(ppo_stats)
        if hasattr(self.repair_trigger_policy, 'training_history'):
            self.training_history.setdefault('repair_trigger_updates', []).append(list(self.repair_trigger_policy.training_history[-step:]) if step > 0 else [])
        return {'episode': episode, 'total_reward': episode_total_reward, 'length': step, 'reward_components': reward_dict, 'weights': self.llm_controller.get_current_weights(), 'ppo_stats': ppo_stats}

    def train(self, num_episodes: Optional[int]=None) -> Dict[str, Any]:
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        print(f'开始训练，共 {num_episodes} 个回合...')
        for episode in range(num_episodes):
            episode_stats = self.train_episode(episode)
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                print(f'回合 {episode + 1}/{num_episodes}, 平均奖励: {avg_reward:.4f}, 当前奖励: {episode_stats['total_reward']:.4f}')
        print('训练完成！')
        return {'training_history': self.training_history, 'final_weights': self.llm_controller.get_current_weights(), 'agent_stats': self.agent.training_stats}

    def _get_initial_state(self) -> np.ndarray:
        if self.env_interface and hasattr(self.env_interface, 'reset'):
            return self.env_interface.reset()
        return np.zeros(self.agent.state_dim, dtype=np.float32)

    def _default_env_step(self, action: int, state: np.ndarray) -> tuple:
        next_state = state.copy()
        done = False
        info = {}
        return (next_state, done, info)

    def save_checkpoint(self, filepath: str):
        import pickle
        checkpoint = {'agent': self.agent, 'llm_controller': self.llm_controller, 'reward_aggregator': self.reward_aggregator, 'training_history': self.training_history, 'config': self.config}
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f'检查点已保存到: {filepath}')

    def load_checkpoint(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        self.agent = checkpoint['agent']
        self.llm_controller = checkpoint['llm_controller']
        self.reward_aggregator = checkpoint['reward_aggregator']
        self.training_history = checkpoint['training_history']
        self.config = checkpoint['config']
        print(f'检查点已从 {filepath} 加载')
