from __future__ import annotations
from typing import Dict, List

def render_control_to_prompt(control: Dict[str, object]) -> str:
    style_tags = ', '.join(control.get('style_tags') or ['cinematic'])
    chords = ', '.join(control.get('chords') or [])
    roles = ', '.join(control.get('track_roles') or [])
    symbolic_summary = control.get('symbolic_summary') or {}
    symbolic_hint = str(symbolic_summary.get('prompt_hint', '')).strip()
    symbolic_clause = f', symbolic guidance: {symbolic_hint}' if symbolic_hint else ''
    render_windows = control.get('render_windows') or []
    window_clause = ''
    if render_windows:
        energy_curve = '-'.join((f'{float(window.get('energy', 0.5)):.2f}' for window in render_windows[:16]))
        brightness_curve = '-'.join((f'{float(window.get('brightness_hint', 0.5)):.2f}' for window in render_windows[:16]))
        window_clause = f', {float(control.get('render_window_seconds', 0.4)):.2f}s rendering-control windows, energy envelope {energy_curve}, brightness envelope {brightness_curve}'
    return f'{style_tags} soundtrack, {control.get('emotion', 'neutral')} mood, {control.get('tempo_bpm', 120)} bpm, {control.get('meter', '4/4')}, key {control.get('key', 'Cmaj')}, section {control.get('section', 'sustain')}, {control.get('texture_hint', 'focused melodic texture')}, track roles: {roles}, chord motion: {chords}{symbolic_clause}{window_clause}'

def render_controls_to_prompts(controls: List[Dict[str, object]]) -> List[str]:
    return [render_control_to_prompt(control) for control in controls]
