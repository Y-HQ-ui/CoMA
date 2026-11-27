from __future__ import annotations
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from .config import RLConfig, build_default_rl_config

def default_config() -> RLConfig:
    return build_default_rl_config()

def build_reward_trainer(*, config: Optional[RLConfig]=None, state_dim: int=64, action_dim: int=32, device: str='cpu', agent: Optional[Any]=None, env_interface: Optional[Callable[[int, np.ndarray], Tuple[np.ndarray, bool, Dict[str, Any]]]]=None, llm_interface: Optional[Callable[..., Dict[str, Any]]]=None) -> Any:
    cfg = config or default_config()
    if agent is None:
        from .agent import RLAgent
        agent = RLAgent(state_dim=state_dim, action_dim=action_dim, learning_rate=cfg.learning_rate, gamma=cfg.gamma, device=device, hidden_dim=cfg.hidden_dim)
    from .trainer import RLTrainer
    return RLTrainer(config=cfg, agent=agent, env_interface=env_interface, llm_interface=llm_interface)

def get_integration_status() -> Dict[str, Any]:
    cfg = default_config()
    return {'ok': True, 'config': asdict(cfg), 'trainer_build': {'supports_build_reward_trainer': True, 'note': '为避免 torch/OpenMP 环境崩溃，这里不默认构建 RLAgent/RLTrainer；需要时可调用 build_reward_trainer()'}, 'critic_repair': {'supports_feedback_source_switch': True, 'supports_rule_based_diagnosis': True, 'supports_selective_regeneration_plan': True, 'supports_repair_execution': True}}
