import numpy as np
import torch
from typing import Dict, Any, Optional
try:
    from .config import build_default_rl_config
    from .agent import RLAgent
    from .coma_env import CoMARLEnv
    from .trainer import RLTrainer
    from .hierarchical_refine import refine_blueprint_hierarchical
    from planning import blueprint_from_segment_priors
except ImportError:
    from config import build_default_rl_config
    from agent import RLAgent
    from coma_env import CoMARLEnv
    from trainer import RLTrainer
    from hierarchical_refine import refine_blueprint_hierarchical
    from planning import blueprint_from_segment_priors

def example_llm_interface(R_vis: float, R_mus: float, R_cons: float, H_human: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    if R_vis < 0.4:
        alpha_vis = np.array([0.4, 0.3, 0.3])
    else:
        alpha_vis = np.array([0.3, 0.3, 0.2])
    if R_mus < 0.4:
        beta_mus = np.array([0.3, 0.3, 0.2, 0.2])
    else:
        beta_mus = np.array([0.25, 0.25, 0.25, 0.25])
    if R_cons < 0.4:
        gamma_cons = 0.4
    else:
        gamma_cons = 0.3
    return {'alpha_vis': alpha_vis, 'beta_mus': beta_mus, 'gamma_cons': gamma_cons}

def build_demo_blueprint():
    sample_prior = {'prior_5d': [{'start_time': 0.0, 'end_time': 4.0, 'motion_intensity': 0.72, 'cut_density': 0.55, 'brightness': 0.38, 'color_warmth': 0.62, 'visual_novelty': 0.58}, {'start_time': 4.0, 'end_time': 8.0, 'motion_intensity': 0.48, 'cut_density': 0.35, 'brightness': 0.52, 'color_warmth': 0.57, 'visual_novelty': 0.44}, {'start_time': 8.0, 'end_time': 12.0, 'motion_intensity': 0.81, 'cut_density': 0.7, 'brightness': 0.66, 'color_warmth': 0.54, 'visual_novelty': 0.69}]}
    return blueprint_from_segment_priors(sample_prior)

def main():
    config = build_default_rl_config()
    blueprint = build_demo_blueprint()
    env = CoMARLEnv(blueprint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = RLAgent(state_dim=env.state_dim, action_dim=env.action_dim, learning_rate=config.learning_rate, gamma=config.gamma, device=device, hidden_dim=config.hidden_dim)
    trainer = RLTrainer(config=config, agent=agent, env_interface=env, llm_interface=example_llm_interface)
    print('=' * 50)
    print('CoMA PPO强化学习训练开始')
    print('=' * 50)
    results = trainer.train(num_episodes=config.num_episodes)
    trainer.save_checkpoint('MASKED_PATH/resource_026.pkl')
    agent.save('MASKED_PATH/resource_027.pth')
    print('\n' + '=' * 50)
    print('训练完成！')
    print('=' * 50)
    print(f'平均奖励: {np.mean(results['training_history']['episode_rewards']):.4f}')
    print(f'最终权重: {results['final_weights']}')
    return results

def demo_hierarchical_main(num_episodes: int=5):
    blueprint = build_demo_blueprint()
    results = refine_blueprint_hierarchical(blueprint, num_episodes=num_episodes)
    print('=' * 50)
    print('CoMA 双层强化学习训练完成')
    print('=' * 50)
    print(f'符号层平均奖励: {np.mean(results['training_history']['symbolic_episode_rewards']):.4f}')
    print(f'音频层平均奖励: {np.mean(results['training_history']['audio_episode_rewards']):.4f}')
    return results
if __name__ == '__main__':
    import os
    os.makedirs('checkpoints', exist_ok=True)
    main()
