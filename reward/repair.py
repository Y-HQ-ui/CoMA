from __future__ import annotations
import copy
from typing import Any, Dict
from audio.abc_to_render_controls import blueprint_to_render_controls
from audio.musicgen_adapter import render_controls_to_prompts
from planning.blueprint_to_abc import blueprint_to_bar_contexts, blueprint_to_mupt_prefixes
from .critic import build_critic_decision
from planning.blueprint_schema import BlueprintTrack, MusicBlueprint

def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))

def _dominant_root(key: str) -> str:
    if key.lower().endswith(('maj', 'min')):
        return key[:-3]
    return key

def _rewrite_chords(chords: list[str], *, key: str, scope: str, deficit: float) -> list[str]:
    base = list(chords) if chords else [key]
    root = _dominant_root(key)
    if scope == 'blueprint':
        if len(base) >= 2:
            return base[:-2] + [f'{root}7', f'{root}m' if key.lower().endswith('min') else root]
        return [f'{root}7', f'{root}m' if key.lower().endswith('min') else root]
    if scope == 'symbolic':
        if deficit > 0.15:
            return [f'{c}7' if '7' not in c else c for c in base]
        return [f'{c}sus2' if 'sus' not in c else c for c in base]
    return base

def _ensure_role(tracks: list[BlueprintTrack], role: str, register: str, phrase_function: str, energy: float) -> list[BlueprintTrack]:
    if any((track.role == role for track in tracks)):
        return tracks
    return tracks + [BlueprintTrack(role, register, _clip01(energy + 0.05), phrase_function)]

def _remove_role(tracks: list[BlueprintTrack], role: str) -> list[BlueprintTrack]:
    filtered = [track for track in tracks if track.role != role]
    return filtered or tracks

def repair_blueprint_deterministic(blueprint: MusicBlueprint, diagnosis: Dict[str, Any], *, scope: str, num_passes: int=1) -> MusicBlueprint:
    repaired = copy.deepcopy(blueprint)
    deficits = diagnosis.get('deficits', {})
    visual_deficit = float(deficits.get('visual', 0.0))
    musicality_deficit = float(deficits.get('musicality', 0.0))
    consistency_deficit = float(deficits.get('consistency', 0.0))
    symbolic_deficit = float(deficits.get('symbolic', 0.0))
    dominant = str(diagnosis.get('dominant_failure', 'symbolic'))
    for _ in range(max(1, num_passes)):
        for idx, segment in enumerate(repaired.segments):
            energy_boost = 0.05 * (visual_deficit + musicality_deficit)
            if scope == 'blueprint':
                segment.section = 'transition' if idx < len(repaired.segments) - 1 else 'climax'
                segment.tempo_bpm = int(max(70, min(170, segment.tempo_bpm + round(6 * visual_deficit))))
                segment.energy = _clip01(segment.energy + energy_boost)
                segment.chords = _rewrite_chords(segment.chords, key=segment.global_key, scope='blueprint', deficit=visual_deficit)
                segment.tracks = _ensure_role(segment.tracks, 'harmony', 'mid', 'support', segment.energy)
                if visual_deficit > 0.15 or dominant == 'visual':
                    segment.tracks = _ensure_role(segment.tracks, 'drums', 'mid', 'rhythmic', segment.energy)
            elif scope == 'symbolic':
                segment.tempo_bpm = int(max(70, min(170, segment.tempo_bpm + round(4 * symbolic_deficit))))
                segment.energy = _clip01(segment.energy + 0.03 * symbolic_deficit)
                segment.chords = _rewrite_chords(segment.chords, key=segment.global_key, scope='symbolic', deficit=symbolic_deficit)
                if symbolic_deficit > 0.1:
                    segment.tracks = _ensure_role(segment.tracks, 'harmony', 'mid', 'support', segment.energy)
                if musicality_deficit > 0.15:
                    segment.tracks = _remove_role(segment.tracks, 'fx')
            elif consistency_deficit > 0.1:
                segment.energy = _clip01(segment.energy - 0.02)
                segment.tempo_bpm = int(max(70, min(170, segment.tempo_bpm - 2)))
    return repaired

def repair_render_controls_deterministic(controls: list[Dict[str, Any]], diagnosis: Dict[str, Any], *, scope: str, num_passes: int=1) -> list[Dict[str, Any]]:
    repaired = [dict(control) for control in controls]
    deficits = diagnosis.get('deficits', {})
    consistency_deficit = float(deficits.get('consistency', 0.0))
    visual_deficit = float(deficits.get('visual', 0.0))
    musicality_deficit = float(deficits.get('musicality', 0.0))
    total_boost = 0.04 * (consistency_deficit + visual_deficit + musicality_deficit)
    for _ in range(max(1, num_passes)):
        for control in repaired:
            if scope == 'audio':
                control['tempo_bpm'] = int(max(70, min(170, control['tempo_bpm'] + round(8 * consistency_deficit))))
                control['energy'] = _clip01(control['energy'] + total_boost)
                control['brightness_hint'] = _clip01(control.get('brightness_hint', control['video_prior_5d']['brightness']) + 0.08 * visual_deficit)
                control['texture_bias'] = float(control.get('texture_bias', 0.0) + 0.05 * musicality_deficit)
            elif scope == 'symbolic':
                control['energy'] = _clip01(control['energy'] + 0.03 * musicality_deficit)
                control['texture_hint'] = f'{control.get('texture_hint', 'focused melodic texture')} with smoother phrasing'
            else:
                control['tempo_bpm'] = int(max(70, min(170, control['tempo_bpm'] + round(4 * consistency_deficit))))
                control['energy'] = _clip01(control['energy'] + 0.02 * visual_deficit)
    return repaired

def execute_repair_cycle(blueprint: MusicBlueprint, reward_dict: Dict[str, float], *, feedback_mode: str, global_threshold: float, component_threshold: float, max_repair_passes: int, human_feedback: Dict[str, Any] | None=None, llm_feedback: Dict[str, Any] | None=None, llm_critic: Any=None, critic_context: Dict[str, Any] | None=None) -> Dict[str, Any]:
    decision = build_critic_decision(reward_dict, feedback_mode=feedback_mode, global_threshold=global_threshold, component_threshold=component_threshold, max_repair_passes=max_repair_passes, human_feedback=human_feedback, llm_feedback=llm_feedback, llm_critic=llm_critic, critic_context=critic_context)
    repair_plan = decision.repair_plan
    repaired_blueprint = blueprint
    repaired_controls = blueprint_to_render_controls(blueprint)
    action_trace: list[Dict[str, Any]] = []
    for action in repair_plan['actions']:
        if action == 'replan_blueprint':
            repaired_blueprint = repair_blueprint_deterministic(repaired_blueprint, decision.diagnosis, scope='blueprint', num_passes=repair_plan['num_passes'])
            repaired_controls = blueprint_to_render_controls(repaired_blueprint)
        elif action == 'rerun_symbolic':
            repaired_blueprint = repair_blueprint_deterministic(repaired_blueprint, decision.diagnosis, scope='symbolic', num_passes=repair_plan['num_passes'])
            repaired_controls = blueprint_to_render_controls(repaired_blueprint)
            repaired_controls = repair_render_controls_deterministic(repaired_controls, decision.diagnosis, scope='symbolic', num_passes=repair_plan['num_passes'])
        elif action == 'rerun_audio':
            repaired_controls = repair_render_controls_deterministic(repaired_controls, decision.diagnosis, scope='audio', num_passes=repair_plan['num_passes'])
        action_trace.append({'action': action, 'num_segments': len(repaired_blueprint.segments), 'num_controls': len(repaired_controls), 'first_segment_chords': list(repaired_blueprint.segments[0].chords) if repaired_blueprint.segments else [], 'first_prompt': render_controls_to_prompts(repaired_controls[:1])[0] if repaired_controls else ''})
    return {'critic_decision': {'feedback_source': decision.feedback_source, 'diagnosis': decision.diagnosis, 'repair_plan': repair_plan}, 'repair_action_trace': action_trace, 'repaired_blueprint': repaired_blueprint.to_dict(), 'repaired_bar_contexts': blueprint_to_bar_contexts(repaired_blueprint), 'repaired_mupt_prefixes': blueprint_to_mupt_prefixes(repaired_blueprint), 'repaired_render_controls': repaired_controls, 'repaired_prompts': render_controls_to_prompts(repaired_controls)}

def summarize_repair_cycle(blueprint: MusicBlueprint, reward_dict: Dict[str, float], *, feedback_mode: str, global_threshold: float, component_threshold: float, max_repair_passes: int, human_feedback: Dict[str, Any] | None=None, llm_feedback: Dict[str, Any] | None=None, llm_critic: Any=None, critic_context: Dict[str, Any] | None=None) -> Dict[str, Any]:
    return execute_repair_cycle(blueprint=blueprint, reward_dict=reward_dict, feedback_mode=feedback_mode, global_threshold=global_threshold, component_threshold=component_threshold, max_repair_passes=max_repair_passes, human_feedback=human_feedback, llm_feedback=llm_feedback, llm_critic=llm_critic, critic_context=critic_context)
