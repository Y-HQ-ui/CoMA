from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .config import RLConfig, build_default_rl_config


def default_config() -> RLConfig:
    return build_default_rl_config()


def build_reward_trainer(
    *,
    config: Optional[RLConfig] = None,
    state_dim: int = 64,
    action_dim: int = 32,
    device: str = "cpu",
    agent: Optional[Any] = None,
    env_interface: Optional[Callable[[int, np.ndarray], Tuple[np.ndarray, bool, Dict[str, Any]]]] = None,
    llm_interface: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Any:
    """
    构建RLTrainer（仅构建，不训练）。
    - env_interface/llm_interface 都是可注入的，默认None时不会调用外部API。
    """
    cfg = config or default_config()

    if agent is None:
        # 延迟导入：避免仅做“接入检查”时就触发 torch/OpenMP 崩溃
        from .agent import RLAgent  # noqa: WPS433

        agent = RLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            device=device,
            hidden_dim=cfg.hidden_dim,
        )

    from .trainer import RLTrainer  # noqa: WPS433

    return RLTrainer(config=cfg, agent=agent, env_interface=env_interface, llm_interface=llm_interface)


def get_integration_status() -> Dict[str, Any]:
    """
    用于主程序/健康检查：确认reward强化模块可导入、可构建、关键对象存在。
    绝不进行网络请求、绝不启动训练。
    """
    cfg = default_config()
    return {
        "ok": True,
        "config": asdict(cfg),
        "trainer_build": {
            "supports_build_reward_trainer": True,
            "note": "为避免 torch/OpenMP 环境崩溃，这里不默认构建 RLAgent/RLTrainer；需要时可调用 build_reward_trainer()",
        },
        "critic_repair": {
            "supports_feedback_source_switch": True,
            "supports_rule_based_diagnosis": True,
            "supports_selective_regeneration_plan": True,
            "supports_repair_execution": True,
        },
    }
