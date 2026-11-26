import numpy as np
import torch
try:
    from .rl_config import RLConfig
    from .rl_agent import RLAgent
    from .rl_trainer import RLTrainer
except ImportError:
    from rl_config import RLConfig
    from rl_agent import RLAgent
    from rl_trainer import RLTrainer


def custom_env_interface(action: int, state: np.ndarray) -> tuple:
    next_state = state + np.random.randn(*state.shape) * 0.05
    done = np.random.rand() < 0.02
    
    info = {
        'symbolic_data': {
            'beat': {
                'g_vis': np.random.rand(100) * 0.8 + 0.1,
                'g_sym': np.random.rand(100) * 0.8 + 0.1
            },
            'phrase': {
                'B_vis': [10, 30, 50, 70, 90],
                'B_sym': [12, 32, 52, 72, 92]
            },
            'key': {
                'key_posterior': np.random.dirichlet(np.ones(12) * 2, size=100)
            },
            'diversity': {
                'pitches': np.random.randint(60, 72, 50),
                'durations': np.random.choice([0.25, 0.5, 1.0, 2.0], 50)
            },
            'playability': {
                'violations': np.random.randint(0, 5),
                'total_notes': 50
            }
        },
        'consistency_data': {
            'onset': {
                'e_sym': np.random.rand(100) * 0.8 + 0.1,
                'e_aud': np.random.rand(100) * 0.8 + 0.1
            },
            'dynamic': {
                'rms_aud': np.random.rand(100) * 0.8 + 0.1,
                'vel_sym': np.random.rand(100) * 0.8 + 0.1
            },
            'color': {
                'C_spec': np.random.rand(100) * 2000 + 1000,
                'w': np.random.rand(100),
                'hf_ratio': np.random.rand(100),
                'b': np.random.rand(100)
            }
        },
        'visual_data': {
            'motion_rhythm': np.random.rand(),
            'color_tone': np.random.rand(),
            'novelty_thematic': np.random.rand()
        },
        'musicality_data': {
            'tonal_stability': np.random.rand(),
            'harmonic_progression': np.random.rand(),
            'rhythmic_density': np.random.rand(),
            'structural_symmetry': np.random.rand()
        },
        'kl_data': {
            'p_y': np.random.dirichlet(np.ones(10)),
            'q_y': np.random.dirichlet(np.ones(10))
        },
        'human_feedback': None
    }
    
    return next_state, done, info


def custom_llm_interface(R_vis: float, R_mus: float, R_cons: float,
                         H_human=None) -> dict:
    if R_vis < 0.4:
        alpha_vis = np.array([0.4, 0.35, 0.25])
    elif R_vis > 0.7:
        alpha_vis = np.array([0.25, 0.25, 0.2])
    else:
        alpha_vis = np.array([0.3, 0.3, 0.2])
    
    if R_mus < 0.4:
        beta_mus = np.array([0.3, 0.3, 0.2, 0.2])
    elif R_mus > 0.7:
        beta_mus = np.array([0.2, 0.2, 0.25, 0.25])
    else:
        beta_mus = np.array([0.25, 0.25, 0.25, 0.25])
    
    if R_cons < 0.4:
        gamma_cons = 0.4
    elif R_cons > 0.7:
        gamma_cons = 0.25
    else:
        gamma_cons = 0.3
    
    if H_human is not None:
        preference = H_human.get('preference', {})
        if preference.get('prefer_visual', False):
            alpha_vis = alpha_vis * 1.2
        if preference.get('prefer_musicality', False):
            beta_mus = beta_mus * 1.2
    
    return {
        'alpha_vis': alpha_vis,
        'beta_mus': beta_mus,
        'gamma_cons': gamma_cons
    }


def main():
    print("=" * 60)
    print("强化学习训练系统示例")
    print("=" * 60)
    
    print("\n[1] 创建配置...")
    config = RLConfig(
        lambda_beat=0.3,
        lambda_phrase=0.25,
        lambda_key=0.2,
        lambda_div=0.15,
        lambda_play=0.1,
        eta_onset=0.4,
        eta_dyn=0.35,
        eta_color=0.25,
        alpha_vis=[0.3, 0.3, 0.2],
        beta_mus=[0.25, 0.25, 0.25, 0.25],
        gamma_cons=0.3,
        tau=0.15,
        epsilon=1e-3,
        delta=0.1,
        llm_update_freq=5,
        num_episodes=50,
        max_steps_per_episode=30,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=32,
        delta_max=0.1,
        temporal_alignment_freq=2.0,
        use_llm_controller=True,
        llm_model_name=None,
        human_feedback_enabled=False
    )
    print("✓ 配置创建完成")
    
    print("\n[2] 创建智能体...")
    state_dim = 64
    action_dim = 32
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    agent = RLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        device=device
    )
    print("✓ 智能体创建完成")
    
    print("\n[3] 创建训练器...")
    trainer = RLTrainer(
        config=config,
        agent=agent,
        env_interface=custom_env_interface,
        llm_interface=custom_llm_interface
    )
    print("✓ 训练器创建完成")
    
    print("\n[4] 开始训练...")
    print("-" * 60)
    
    results = trainer.train(num_episodes=config.num_episodes)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    episode_rewards = results['training_history']['episode_rewards']
    print(f"\n统计信息:")
    print(f"  总回合数: {len(episode_rewards)}")
    print(f"  平均奖励: {np.mean(episode_rewards):.4f}")
    print(f"  最大奖励: {np.max(episode_rewards):.4f}")
    print(f"  最小奖励: {np.min(episode_rewards):.4f}")
    print(f"  最后10回合平均: {np.mean(episode_rewards[-10:]):.4f}")
    
    print(f"\n最终权重:")
    final_weights = results['final_weights']
    print(f"  Alpha (视觉): {final_weights['alpha_vis']}")
    print(f"  Beta (音乐性): {final_weights['beta_mus']}")
    print(f"  Gamma (一致性): {final_weights['gamma_cons']:.4f}")
    
    print("\n[5] 保存模型...")
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    trainer.save_checkpoint('checkpoints/rl_training_checkpoint.pkl')
    agent.save('checkpoints/rl_agent_model.pth')
    print("✓ 模型保存完成")
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    results = main()

