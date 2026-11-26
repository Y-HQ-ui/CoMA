import numpy as np
from typing import Dict, List, Optional, Tuple, Any
try:
    from .config import RLConfig
except ImportError:
    from config import RLConfig


class LLMWeightController:
    def __init__(self, config: RLConfig, llm_interface=None):
        self.config = config
        self.llm_interface = llm_interface

        self.alpha_vis, self.beta_mus, self.gamma_cons = self._normalize_joint(
            np.array(config.alpha_vis, dtype=np.float32),
            np.array(config.beta_mus, dtype=np.float32),
            float(config.gamma_cons),
        )
        
        self.weight_history = []
        self.reward_history = []
        self.feedback_history = []
    
    def softmax_normalize(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = exp_logits / exp_logits.sum()
        
        weights = np.clip(weights, self.config.epsilon, 1.0)
        weights = weights / weights.sum()
        
        return weights

    def _normalize_joint(
        self,
        alpha_vis: np.ndarray,
        beta_mus: np.ndarray,
        gamma_cons: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        from .weight_constraints import simplex_with_floor
        alpha_vis = np.asarray(alpha_vis, dtype=np.float64).reshape(-1)
        beta_mus = np.asarray(beta_mus, dtype=np.float64).reshape(-1)
        scores = np.concatenate([alpha_vis, beta_mus, [gamma_cons]])
        normalized = simplex_with_floor(scores,self.config.epsilon,self.config.weight_temperature)
        alpha_end = len(alpha_vis)
        beta_end = alpha_end + len(beta_mus)
        return normalized[:alpha_end], normalized[alpha_end:beta_end], float(normalized[-1])

    def _coerce_proposal(
        self,
        suggestions: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        alpha_vis = np.asarray(suggestions.get("alpha_vis", self.alpha_vis), dtype=np.float32).reshape(-1)
        beta_mus = np.asarray(suggestions.get("beta_mus", self.beta_mus), dtype=np.float32).reshape(-1)
        if alpha_vis.size != self.alpha_vis.size:
            alpha_vis = self.alpha_vis.copy()
        if beta_mus.size != self.beta_mus.size:
            beta_mus = self.beta_mus.copy()
        gamma_cons = float(suggestions.get("gamma_cons", self.gamma_cons))
        return self._normalize_joint(alpha_vis, beta_mus, gamma_cons)
    
    def infer_weights_llm(self,
                         R_vis: float,
                         R_mus: float,
                         R_cons: float,
                         H_human: Optional[Dict[str, Any]] = None,
                         H_llm: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        base = {
            "alpha_vis": self.alpha_vis,
            "beta_mus": self.beta_mus,
            "gamma_cons": self.gamma_cons,
        }
        if self.llm_interface is not None:
            try:
                suggestions = self.llm_interface(
                    R_vis=R_vis,
                    R_mus=R_mus,
                    R_cons=R_cons,
                    H_human=H_human,
                    H_llm=H_llm,
                )
                if isinstance(suggestions, dict) and suggestions:
                    return {
                        "alpha_vis": np.array(suggestions.get("alpha_vis", base["alpha_vis"]), dtype=np.float32),
                        "beta_mus": np.array(suggestions.get("beta_mus", base["beta_mus"]), dtype=np.float32),
                        "gamma_cons": float(suggestions.get("gamma_cons", base["gamma_cons"])),
                    }
            except Exception as e:
                print(f"LLM接口调用失败，使用规则化反馈: {e}")

        structured_feedback = H_llm or H_human
        if structured_feedback:
            return self._weights_from_structured_feedback(structured_feedback, fallback=base)
        return self._heuristic_weight_adjustment(R_vis, R_mus, R_cons)

    def _weights_from_structured_feedback(
        self,
        feedback: Dict[str, Any],
        *,
        fallback: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        alpha_vis = np.array(feedback.get("alpha_vis", fallback["alpha_vis"]), dtype=np.float32)
        beta_mus = np.array(feedback.get("beta_mus", fallback["beta_mus"]), dtype=np.float32)
        gamma_cons = float(feedback.get("gamma_cons", fallback["gamma_cons"]))

        focus = str(feedback.get("focus", "")).lower()
        score = float(feedback.get("score", 0.5))
        strength = float(feedback.get("strength", self.config.human_feedback_weight))
        gain = max(0.0, min(1.0, (1.0 - score) * strength))

        if focus in {"alignment", "visual", "vision"}:
            alpha_vis = alpha_vis * (1.0 + 0.4 * gain)
        elif focus in {"musicality", "music"}:
            beta_mus = beta_mus * (1.0 + 0.4 * gain)
        elif focus in {"consistency", "audio"}:
            gamma_cons = gamma_cons * (1.0 + 0.4 * gain)

        return {
            "alpha_vis": alpha_vis,
            "beta_mus": beta_mus,
            "gamma_cons": gamma_cons,
        }
    
    def _heuristic_weight_adjustment(self,
                                    R_vis: float,
                                    R_mus: float,
                                    R_cons: float) -> Dict[str, np.ndarray]:
        if R_vis < 0.5:
            alpha_vis_new = self.alpha_vis * (1.0 + 0.1 * (1.0 - R_vis))
        else:
            alpha_vis_new = self.alpha_vis * (1.0 - 0.05 * R_vis)
        
        if R_mus < 0.5:
            beta_mus_new = self.beta_mus * (1.0 + 0.1 * (1.0 - R_mus))
        else:
            beta_mus_new = self.beta_mus * (1.0 - 0.05 * R_mus)
        
        if R_cons < 0.5:
            gamma_cons_new = self.gamma_cons * (1.0 + 0.1 * (1.0 - R_cons))
        else:
            gamma_cons_new = self.gamma_cons * (1.0 - 0.05 * R_cons)
        
        return {
            'alpha_vis': alpha_vis_new,
            'beta_mus': beta_mus_new,
            'gamma_cons': gamma_cons_new
        }
    
    def update_weights(self,
                      R_vis: float,
                      R_mus: float,
                      R_cons: float,
                      H_human: Optional[Dict[str, Any]] = None,
                      H_llm: Optional[Dict[str, Any]] = None,
                      step: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
        self.reward_history.append({
            'R_vis': R_vis,
            'R_mus': R_mus,
            'R_cons': R_cons,
            'step': step
        })

        feedback_mode = getattr(self.config, "critic_feedback_mode", "llm")
        base = {
            'alpha_vis': self.alpha_vis,
            'beta_mus': self.beta_mus,
            'gamma_cons': self.gamma_cons
        }
        if "human" in feedback_mode and self.config.human_feedback_enabled and H_human:
            suggestions = self._weights_from_structured_feedback(H_human, fallback=base)
            feedback_source = "human"
        elif self.config.weight_mode == "deterministic" and step % self.config.llm_update_freq == 0:
            suggestions = self._heuristic_weight_adjustment(R_vis, R_mus, R_cons)
            feedback_source = "deterministic"
        elif self.config.weight_mode != "fixed" and step % self.config.llm_update_freq == 0 and self.config.use_llm_controller:
            suggestions = self.infer_weights_llm(R_vis, R_mus, R_cons, H_human, H_llm)
            feedback_source = "llm_or_heuristic"
        else:
            suggestions = base
            feedback_source = "static"
        
        if feedback_source == "static":
            proposed_alpha, proposed_beta, proposed_gamma = self.alpha_vis.copy(), self.beta_mus.copy(), self.gamma_cons
        else:
            proposed_alpha, proposed_beta, proposed_gamma = self._coerce_proposal(suggestions)
        alpha_vis_new = (1 - self.config.tau) * self.alpha_vis + self.config.tau * proposed_alpha
        beta_mus_new = (1 - self.config.tau) * self.beta_mus + self.config.tau * proposed_beta
        gamma_cons_new = (1 - self.config.tau) * self.gamma_cons + self.config.tau * proposed_gamma

        combined_delta = np.concatenate([
            alpha_vis_new - self.alpha_vis,
            beta_mus_new - self.beta_mus,
            np.array([gamma_cons_new - self.gamma_cons]),
        ])
        delta_norm = float(np.linalg.norm(combined_delta))

        if delta_norm > self.config.delta:
            scale = self.config.delta / (delta_norm + 1e-8)
            alpha_vis_new = self.alpha_vis + scale * (alpha_vis_new - self.alpha_vis)
            beta_mus_new = self.beta_mus + scale * (beta_mus_new - self.beta_mus)
            gamma_cons_new = self.gamma_cons + scale * (gamma_cons_new - self.gamma_cons)

        # Both endpoints lie on the same floored simplex. Smoothing and the
        # L2 step bound preserve it; re-projecting here would move fixed weights.
        
        self.alpha_vis = alpha_vis_new
        self.beta_mus = beta_mus_new
        self.gamma_cons = gamma_cons_new
        
        self.weight_history.append({
            'alpha_vis': alpha_vis_new.copy(),
            'beta_mus': beta_mus_new.copy(),
            'gamma_cons': gamma_cons_new,
            'step': step
        })
        self.feedback_history.append({
            'step': step,
            'feedback_source': feedback_source,
            'human_feedback': H_human,
            'llm_feedback': H_llm,
        })
        
        return alpha_vis_new, beta_mus_new, gamma_cons_new
    
    def get_current_weights(self) -> Dict[str, Any]:
        return {
            'alpha_vis': self.alpha_vis.copy(),
            'beta_mus': self.beta_mus.copy(),
            'gamma_cons': self.gamma_cons
        }
    
    def reset(self):
        self.alpha_vis, self.beta_mus, self.gamma_cons = self._normalize_joint(
            np.array(self.config.alpha_vis, dtype=np.float32),
            np.array(self.config.beta_mus, dtype=np.float32),
            float(self.config.gamma_cons),
        )
        self.weight_history = []
        self.reward_history = []
        self.feedback_history = []
