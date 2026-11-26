from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CriticDecision:
    feedback_source: str
    diagnosis: Dict[str, Any]
    repair_plan: Dict[str, Any]


def choose_feedback_source(
    *,
    mode: str,
    human_feedback: Optional[Dict[str, Any]] = None,
    llm_feedback: Optional[Dict[str, Any]] = None,
) -> tuple[str, Optional[Dict[str, Any]]]:
    mode = (mode or "llm").lower()
    has_human = bool(human_feedback)
    has_llm = bool(llm_feedback)

    if has_human and "human" in mode:
        return "human", human_feedback
    if mode == "llm" and has_llm:
        return "llm", llm_feedback
    if mode == "human_or_llm":
        if has_llm:
            return "llm", llm_feedback
    if mode == "llm_or_human":
        if has_llm:
            return "llm", llm_feedback
    return "rule", None


def diagnose_failure(
    reward_dict: Dict[str, float],
    *,
    global_threshold: float,
    component_threshold: float,
) -> Dict[str, Any]:
    from .paper_objectives import AUDIO_NAMES
    if all(name in reward_dict for name in AUDIO_NAMES):
        # Minimum aggregation preserves the largest individual deficit. The
        # category mapping is explicit because the paper gives no per-term map.
        components = {
            "visual": min(float(reward_dict[n]) for n in ("motion", "cut", "emotion")),
            "musicality": min(float(reward_dict[n]) for n in ("tonal", "novelty")),
            "consistency": float(reward_dict["consistency"]),
        }
    else:
        components = {
            "visual": float(reward_dict.get("R_vis", 0.0)),
            "musicality": float(reward_dict.get("R_mus", 0.0)),
            "consistency": float(reward_dict.get("R_cons", 0.0)),
            "symbolic": float(reward_dict.get("R_sym", reward_dict.get("R_mus", 0.0))),
        }
    deficits = {name: max(0.0, component_threshold - value) for name, value in components.items()}
    dominant = max(deficits, key=deficits.get)

    if dominant == "consistency":
        source_level = "audio"
    elif dominant == "symbolic" or dominant == "musicality":
        source_level = "symbolic"
    elif dominant == "visual":
        source_level = "blueprint"
    else:
        source_level = "audio"

    global_reward = float(reward_dict.get("R_total", 0.0))
    repair_required = bool(global_reward < global_threshold or deficits[dominant] > 0.0)
    return {
        "repair_required": repair_required,
        "global_reward": global_reward,
        "dominant_failure": dominant,
        "source_level": source_level,
        "deficits": deficits,
    }


def make_repair_plan(
    diagnosis: Dict[str, Any],
    *,
    max_passes: int,
) -> Dict[str, Any]:
    if not diagnosis.get("repair_required", False):
        return {
            "scope": "none",
            "num_passes": 0,
            "actions": [],
        }

    scope = diagnosis["source_level"]
    if scope == "blueprint":
        actions = ["replan_blueprint", "rerun_symbolic", "rerun_audio"]
    elif scope == "symbolic":
        actions = ["rerun_symbolic", "rerun_audio"]
    else:
        actions = ["rerun_audio"]
    return {
        "scope": scope,
        "num_passes": max_passes,
        "actions": actions,
    }


def build_critic_decision(
    reward_dict: Dict[str, float],
    *,
    feedback_mode: str,
    global_threshold: float,
    component_threshold: float,
    max_repair_passes: int,
    human_feedback: Optional[Dict[str, Any]] = None,
    llm_feedback: Optional[Dict[str, Any]] = None,
    llm_critic: Any = None,
    critic_context: Optional[Dict[str, Any]] = None,
) -> CriticDecision:
    generated_llm_feedback = llm_feedback
    llm_error: Optional[str] = None
    mode = (feedback_mode or "llm").lower()
    if generated_llm_feedback is None and llm_critic is not None and "llm" in mode:
        try:
            generated_llm_feedback = llm_critic.diagnose(
                reward_dict=reward_dict,
                context=critic_context,
            )
        except Exception as exc:
            generated_llm_feedback = None
            llm_error = str(exc)
    feedback_source, feedback_payload = choose_feedback_source(
        mode=feedback_mode,
        human_feedback=human_feedback,
        llm_feedback=generated_llm_feedback,
    )
    diagnosis = diagnose_failure(
        reward_dict,
        global_threshold=global_threshold,
        component_threshold=component_threshold,
    )
    if llm_error:
        diagnosis["llm_error"] = llm_error
        diagnosis["llm_fallback"] = "rule_based"
    if feedback_source == "llm" and feedback_payload:
        llm_deficits = feedback_payload.get("deficits")
        if isinstance(llm_deficits, dict):
            for name in diagnosis["deficits"]:
                if name in llm_deficits:
                    diagnosis["deficits"][name] = max(0.0, float(llm_deficits[name]))
        source_level = feedback_payload.get("source_level")
        if source_level in {"blueprint", "symbolic", "audio"}:
            diagnosis["source_level"] = source_level
        if "repair_required" in feedback_payload:
            diagnosis["repair_required"] = bool(feedback_payload["repair_required"])
        if feedback_payload.get("dominant_failure"):
            diagnosis["dominant_failure"] = str(feedback_payload["dominant_failure"])
        diagnosis["llm_confidence"] = float(feedback_payload.get("confidence", 0.5))
        diagnosis["llm_rationale"] = str(feedback_payload.get("rationale", ""))
        if feedback_payload.get("critic_error"):
            diagnosis["llm_error"] = feedback_payload["critic_error"]
    if feedback_source == "human" and feedback_payload:
        preferred_scope = feedback_payload.get("preferred_scope")
        force_scope = feedback_payload.get("force_scope")
        veto = bool(feedback_payload.get("veto", False))
        if preferred_scope in {"blueprint", "symbolic", "audio"}:
            diagnosis["source_level"] = preferred_scope
        if force_scope in {"blueprint", "symbolic", "audio"}:
            diagnosis["source_level"] = force_scope
            diagnosis["repair_required"] = True
        if "repair_required" in feedback_payload:
            diagnosis["repair_required"] = bool(feedback_payload["repair_required"])
        if veto:
            diagnosis["repair_required"] = False

    repair_plan = make_repair_plan(diagnosis, max_passes=max_repair_passes)
    return CriticDecision(
        feedback_source=feedback_source,
        diagnosis=diagnosis,
        repair_plan=repair_plan,
    )
