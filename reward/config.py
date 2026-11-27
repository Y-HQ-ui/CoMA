from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import numpy as np

def public_config(config):
    from dataclasses import asdict
    return {key: '[redacted]' if 'api_key' in key and value else value for key, value in asdict(config).items()}

@dataclass
class RLConfig:
    lambda_beat: float
    lambda_phrase: float
    lambda_key: float
    lambda_div: float
    lambda_play: float
    eta_onset: float
    eta_dyn: float
    eta_color: float
    alpha_vis: List[float]
    beta_mus: List[float]
    gamma_cons: float
    tau: float
    epsilon: float
    delta: float
    llm_update_freq: int
    num_episodes: int
    max_steps_per_episode: int
    learning_rate: float
    gamma: float
    batch_size: int
    delta_max: float
    temporal_alignment_freq: float
    use_llm_controller: bool
    llm_model_name: Optional[str]
    human_feedback_enabled: bool
    critic_feedback_mode: str
    human_feedback_weight: float
    repair_global_threshold: float
    repair_component_threshold: float
    max_repair_passes: int
    ppo_clip: float
    gae_lambda: float
    rollout_size: int
    update_epochs: int
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    hidden_dim: int
    action_temperature: float
    repair_trigger_base: float
    repair_trigger_deficit_weight: float
    repair_trigger_uncertainty_weight: float
    repair_trigger_hidden_dim: int
    repair_trigger_lr: float
    repair_trigger_blend_alpha: float
    symbolic_warmup_episodes: int = 20
    update_symbolic_during_joint: bool = False
    reward_backend: str = 'auto'
    llm_critic_enabled: bool = False
    llm_critic_base_url: Optional[str] = None
    llm_critic_api_key: Optional[str] = None
    llm_critic_model: Optional[str] = None
    llm_critic_timeout: float = 120.0
    symbolic_gamma: float = 0.95
    audio_gamma: float = 0.8
    audio_window_seconds: float = 0.4
    gradient_free_logit_noise: float = 0.35
    gradient_free_noise_decay: float = 0.85
    consistency_weight_ramp: float = 0.5
    weight_temperature: float = 1.0
    weight_mode: str = 'legacy'
    consistency_enabled: bool = True

    def get_symbolic_weights(self) -> Dict[str, float]:
        return {'beat': self.lambda_beat, 'phrase': self.lambda_phrase, 'key': self.lambda_key, 'div': self.lambda_div, 'play': self.lambda_play}

    def get_consistency_weights(self) -> Dict[str, float]:
        return {'onset': self.eta_onset, 'dyn': self.eta_dyn, 'color': self.eta_color}

    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights, self.epsilon, 1.0)
        weights = weights / weights.sum()
        return weights

def build_default_rl_config() -> RLConfig:
    return RLConfig(lambda_beat=0.3, lambda_phrase=0.25, lambda_key=0.2, lambda_div=0.15, lambda_play=0.1, eta_onset=0.4, eta_dyn=0.35, eta_color=0.25, alpha_vis=[0.4, 0.35, 0.25], beta_mus=[0.3, 0.25, 0.2, 0.25], gamma_cons=0.3, tau=0.1, epsilon=0.0001, delta=0.1, llm_update_freq=8, num_episodes=200, max_steps_per_episode=32, learning_rate=0.0003, gamma=0.99, batch_size=64, delta_max=0.1, temporal_alignment_freq=2.0, use_llm_controller=False, llm_model_name=None, human_feedback_enabled=False, critic_feedback_mode='llm', human_feedback_weight=1.0, repair_global_threshold=0.55, repair_component_threshold=0.45, max_repair_passes=2, ppo_clip=0.2, gae_lambda=0.95, rollout_size=256, update_epochs=4, entropy_coef=0.01, value_coef=0.5, max_grad_norm=1.0, hidden_dim=256, action_temperature=1.0, repair_trigger_base=0.18, repair_trigger_deficit_weight=1.25, repair_trigger_uncertainty_weight=0.65, repair_trigger_hidden_dim=64, repair_trigger_lr=0.001, repair_trigger_blend_alpha=0.6, symbolic_warmup_episodes=20, update_symbolic_during_joint=False, reward_backend='auto', llm_critic_enabled=bool(os.getenv('MUSICAGENT_LLM_CRITIC_MODEL')), llm_critic_base_url=os.getenv('MUSICAGENT_LLM_CRITIC_BASE_URL') or os.getenv('MUSICAGENT_BASE_URL'), llm_critic_api_key=os.getenv('MUSICAGENT_LLM_CRITIC_API_KEY') or os.getenv('MUSICAGENT_API_KEY'), llm_critic_model=os.getenv('MUSICAGENT_LLM_CRITIC_MODEL') or os.getenv('MUSICAGENT_MODEL_NAME'), llm_critic_timeout=float(os.getenv('MUSICAGENT_LLM_CRITIC_TIMEOUT', '120')), symbolic_gamma=0.95, audio_gamma=0.8, audio_window_seconds=0.4, gradient_free_logit_noise=0.35, gradient_free_noise_decay=0.85, consistency_weight_ramp=0.5)
