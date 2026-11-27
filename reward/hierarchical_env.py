from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from audio.abc_to_render_controls import blueprint_to_render_controls
from planning.blueprint_schema import BlueprintTrack, MusicBlueprint
from .evidence import RewardEvidenceEvaluator

def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

@dataclass(frozen=True)
class SymbolicActionSpec:
    name: str
    tempo_delta: int = 0
    energy_delta: float = 0.0
    density_delta: float = 0.0
    add_drums: bool = False
    drop_pad: bool = False
    section_bias: str | None = None
    chord_mode: str = 'keep'
    key_mode: str = 'keep'
    add_role: str | None = None
    remove_role: str | None = None

@dataclass(frozen=True)
class AudioActionSpec:
    name: str
    tempo_delta: int = 0
    energy_delta: float = 0.0
    brightness_delta: float = 0.0
    texture_bias: float = 0.0
DEFAULT_SYMBOLIC_ACTIONS: List[SymbolicActionSpec] = [SymbolicActionSpec('keep_symbolic'), SymbolicActionSpec('push_phrase', tempo_delta=6, energy_delta=0.08, density_delta=0.1, section_bias='climax'), SymbolicActionSpec('pull_phrase', tempo_delta=-6, energy_delta=-0.08, density_delta=-0.1, section_bias='release'), SymbolicActionSpec('densify_tracks', density_delta=0.14), SymbolicActionSpec('simplify_tracks', density_delta=-0.14), SymbolicActionSpec('add_rhythm_track', add_drums=True, add_role='drums', density_delta=0.08), SymbolicActionSpec('drop_pad_track', drop_pad=True, remove_role='pad', density_delta=-0.05), SymbolicActionSpec('transition_boost', tempo_delta=4, energy_delta=0.06, density_delta=0.06, section_bias='transition'), SymbolicActionSpec('reharmonize_relative', chord_mode='relative'), SymbolicActionSpec('cadence_push', chord_mode='cadence', section_bias='transition'), SymbolicActionSpec('open_texture', add_role='fx', remove_role='pad', density_delta=-0.04), SymbolicActionSpec('thicken_harmony', add_role='harmony', chord_mode='support', density_delta=0.08)]
DEFAULT_AUDIO_ACTIONS: List[AudioActionSpec] = [AudioActionSpec('keep_audio'), AudioActionSpec('tempo_up', tempo_delta=8, energy_delta=0.05, texture_bias=0.05), AudioActionSpec('tempo_down', tempo_delta=-8, energy_delta=-0.05, texture_bias=-0.05), AudioActionSpec('energy_up', energy_delta=0.1, brightness_delta=0.04, texture_bias=0.1), AudioActionSpec('energy_down', energy_delta=-0.1, brightness_delta=-0.04, texture_bias=-0.1), AudioActionSpec('brighten_mix', brightness_delta=0.1, texture_bias=0.05), AudioActionSpec('darken_mix', brightness_delta=-0.1, texture_bias=-0.05), AudioActionSpec('accent_render', tempo_delta=4, energy_delta=0.08, texture_bias=0.12)]

class HierarchicalCoMAEnv:

    def __init__(self, blueprint: MusicBlueprint, symbolic_actions: List[SymbolicActionSpec] | None=None, audio_actions: List[AudioActionSpec] | None=None, reward_evaluator: RewardEvidenceEvaluator | None=None, audio_window_seconds: float=0.4):
        self.base_blueprint = blueprint
        self.symbolic_actions = symbolic_actions or DEFAULT_SYMBOLIC_ACTIONS
        self.audio_actions = audio_actions or DEFAULT_AUDIO_ACTIONS
        self.reward_evaluator = reward_evaluator or RewardEvidenceEvaluator()
        self.audio_window_seconds = max(0.2, min(0.5, float(audio_window_seconds)))
        self.index = 0
        self.current_blueprint: MusicBlueprint | None = None
        self.current_controls: List[Dict[str, Any]] = []
        self.base_controls = blueprint_to_render_controls(blueprint, window_seconds=self.audio_window_seconds)
        self.symbolic_action_history: List[Dict[str, Any]] = []
        self.audio_action_history: List[Dict[str, Any]] = []

    @property
    def symbolic_state_dim(self) -> int:
        return 22

    @property
    def audio_state_dim(self) -> int:
        return 18

    @property
    def symbolic_action_dim(self) -> int:
        return len(self.symbolic_actions)

    @property
    def audio_action_dim(self) -> int:
        return len(self.audio_actions)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.index = 0
        self.current_blueprint = copy.deepcopy(self.base_blueprint)
        self.current_controls = blueprint_to_render_controls(self.current_blueprint, window_seconds=self.audio_window_seconds)
        self.symbolic_action_history = []
        self.audio_action_history = []
        return (self.get_symbolic_state(), self.get_audio_state())

    def get_symbolic_state(self) -> np.ndarray:
        segment = self.current_blueprint.segments[self.index]
        prior = segment.video_prior_5d
        track_roles = [track.role for track in segment.tracks]
        avg_density = float(np.mean([track.density for track in segment.tracks])) if segment.tracks else 0.0
        chord_complexity = self._chord_complexity(segment.chords)
        track_count = len(segment.tracks) / 6.0
        minor_mode = 1.0 if segment.global_key.lower().endswith('min') else 0.0
        state = np.array([prior.get('motion_intensity', 0.5), prior.get('cut_density', 0.5), prior.get('brightness', 0.5), prior.get('color_warmth', 0.5), prior.get('visual_novelty', 0.5), segment.energy, segment.tempo_bpm / 180.0, avg_density, len(segment.chords) / 4.0, chord_complexity, track_count, 1.0 if 'melody' in track_roles else 0.0, 1.0 if 'bass' in track_roles else 0.0, 1.0 if 'pad' in track_roles else 0.0, 1.0 if 'drums' in track_roles else 0.0, 1.0 if 'harmony' in track_roles else 0.0, 1.0 if 'fx' in track_roles else 0.0, minor_mode, self.index / max(1, len(self.current_blueprint.segments) - 1), segment.start_time / max(1e-06, self.current_blueprint.segments[-1].end_time), segment.end_time / max(1e-06, self.current_blueprint.segments[-1].end_time), 1.0 if segment.section in {'transition', 'climax'} else 0.0], dtype=np.float32)
        return state

    def get_audio_state(self, window_index: int=0) -> np.ndarray:
        control = self.current_controls[self.index]
        prior = control['video_prior_5d']
        render_windows = control.get('render_windows') or []
        if render_windows:
            resolved_window_index = min(max(0, window_index), len(render_windows) - 1)
            render_window = render_windows[resolved_window_index]
        else:
            resolved_window_index = 0
            render_window = control
        section_map = {'intro': 0.1, 'build': 0.3, 'sustain': 0.5, 'transition': 0.65, 'chorus': 0.7, 'climax': 0.9, 'release': 0.35, 'outro': 0.2}
        return np.array([prior['motion_intensity'], prior['cut_density'], prior['brightness'], prior['color_warmth'], prior['visual_novelty'], control['energy'], control['tempo_bpm'] / 180.0, 1.0 if control['meter'] == '4/4' else 0.5, section_map.get(control['section'], 0.4), 1.0 if 'melody' in control['track_roles'] else 0.0, 1.0 if 'bass' in control['track_roles'] else 0.0, 1.0 if 'pad' in control['track_roles'] else 0.0, len(control['chords']) / 4.0, self.index / max(1, len(self.current_controls) - 1), (len(self.current_controls) - self.index) / max(1, len(self.current_controls)), render_window.get('texture_bias', control.get('texture_bias', 0.0)) + 0.5, render_window['start_time'] / max(1e-06, self.current_controls[-1]['end_time']), render_window['end_time'] / max(1e-06, self.current_controls[-1]['end_time'])], dtype=np.float32)

    def step(self, symbolic_action: int, audio_action: int | List[int]) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
        symbolic_spec = self.symbolic_actions[symbolic_action]
        self._apply_symbolic_action(symbolic_spec)
        audio_specs = self._apply_audio_actions(audio_action)
        reward_info = self.reward_evaluator.evaluate(self.current_blueprint, self.index, self.current_controls[self.index])
        symbolic_reward = _clip01(float(np.mean(list(reward_info['musicality_data'].values()))))
        audio_reward = _clip01(float(np.mean(list(reward_info['visual_data'].values()))))
        reward_info['R_sym_level'] = symbolic_reward
        reward_info['R_aud_level'] = audio_reward
        self.index += 1
        done = self.index >= len(self.current_controls)
        if done:
            next_symbolic_state = np.zeros(self.symbolic_state_dim, dtype=np.float32)
            next_audio_state = np.zeros(self.audio_state_dim, dtype=np.float32)
        else:
            next_symbolic_state = self.get_symbolic_state()
            next_audio_state = self.get_audio_state()
        info = {'symbolic_action_name': symbolic_spec.name, 'audio_action_name': audio_specs[0].name if len(audio_specs) == 1 else 'window_sequence', 'audio_action_names': [spec.name for spec in audio_specs], 'symbolic_segment': self.current_blueprint.segments[self.index - 1].to_dict(), 'render_control': dict(self.current_controls[self.index - 1]), 'reward_breakdown': reward_info, 'symbolic_data': reward_info['symbolic_data'], 'consistency_data': reward_info['consistency_data'], 'visual_data': reward_info['visual_data'], 'musicality_data': reward_info['musicality_data'], 'evidence_source': reward_info.get('evidence_source', {})}
        return (next_symbolic_state, next_audio_state, symbolic_reward, audio_reward, done, info)

    def _apply_symbolic_action(self, spec: SymbolicActionSpec) -> None:
        segment = self.current_blueprint.segments[self.index]
        segment.tempo_bpm = int(np.clip(segment.tempo_bpm + spec.tempo_delta, 70, 170))
        segment.energy = _clip01(segment.energy + spec.energy_delta)
        if spec.section_bias is not None:
            segment.section = spec.section_bias
        for track in segment.tracks:
            track.density = _clip01(track.density + spec.density_delta)
        if spec.chord_mode != 'keep':
            segment.chords = self._rewrite_chords(segment.chords or [segment.global_key], mode=spec.chord_mode, key=segment.global_key, energy=segment.energy)
        if spec.key_mode != 'keep':
            segment.global_key = self._rewrite_key(segment.global_key, spec.key_mode)
        if spec.add_drums:
            self._ensure_role(segment, 'drums', 'mid', 'rhythmic')
        if spec.drop_pad:
            self._remove_role(segment, 'pad')
        if spec.add_role:
            self._ensure_role(segment, spec.add_role, self._default_register(spec.add_role), self._default_phrase_function(spec.add_role))
        if spec.remove_role:
            self._remove_role(segment, spec.remove_role)
        self.current_controls = blueprint_to_render_controls(self.current_blueprint, window_seconds=self.audio_window_seconds)
        self.symbolic_action_history.append({'segment_id': segment.segment_id, 'action': spec.name, 'section': segment.section, 'global_key': segment.global_key, 'chords': list(segment.chords), 'track_roles': [track.role for track in segment.tracks]})

    def _apply_audio_actions(self, actions: int | List[int]) -> List[AudioActionSpec]:
        control = self.current_controls[self.index]
        render_windows = control.get('render_windows') or [control]
        action_indices = [actions] if isinstance(actions, int) else list(actions)
        if not action_indices:
            action_indices = [0]
        if len(action_indices) == 1 and len(render_windows) > 1:
            action_indices = action_indices * len(render_windows)
        if len(action_indices) != len(render_windows):
            raise ValueError('audio action count must match render window count')
        specs = [self.audio_actions[index] for index in action_indices]
        for window_index, (render_window, spec) in enumerate(zip(render_windows, specs)):
            render_window['tempo_bpm'] = int(np.clip(control['tempo_bpm'] + spec.tempo_delta, 70, 170))
            render_window['energy'] = _clip01(control['energy'] + spec.energy_delta)
            render_window['brightness_hint'] = _clip01(control['video_prior_5d']['brightness'] + spec.brightness_delta)
            render_window['texture_bias'] = spec.texture_bias
            self.audio_action_history.append({'segment_id': control['segment_id'], 'window_index': window_index, 'start_time': render_window['start_time'], 'end_time': render_window['end_time'], 'action': spec.name, 'tempo_bpm': render_window['tempo_bpm'], 'energy': render_window['energy'], 'texture_bias': render_window['texture_bias']})
        control['tempo_bpm'] = int(round(np.mean([window['tempo_bpm'] for window in render_windows])))
        control['energy'] = float(np.mean([window['energy'] for window in render_windows]))
        control['brightness_hint'] = float(np.mean([window['brightness_hint'] for window in render_windows]))
        control['texture_bias'] = float(np.mean([window['texture_bias'] for window in render_windows]))
        return specs

    @staticmethod
    def _default_register(role: str) -> str:
        if role == 'bass':
            return 'low'
        if role in {'fx', 'arpeggio'}:
            return 'high'
        if role == 'pad':
            return 'full'
        return 'mid'

    @staticmethod
    def _default_phrase_function(role: str) -> str:
        mapping = {'melody': 'lead', 'bass': 'support', 'drums': 'rhythmic', 'pad': 'texture', 'harmony': 'support', 'fx': 'accent', 'arpeggio': 'motion'}
        return mapping.get(role, 'support')

    def _ensure_role(self, segment, role: str, register: str, phrase_function: str) -> None:
        roles = {track.role for track in segment.tracks}
        if role not in roles:
            segment.tracks.append(BlueprintTrack(role, register, _clip01(segment.energy + 0.05), phrase_function))

    def _remove_role(self, segment, role: str) -> None:
        filtered = [track for track in segment.tracks if track.role != role]
        if filtered:
            segment.tracks = filtered

    @staticmethod
    def _chord_complexity(chords: List[str]) -> float:
        if not chords:
            return 0.0
        rich = sum((1 for chord in chords if any((token in chord for token in ('7', '9', 'sus', 'dim', 'aug', 'maj7')))))
        return rich / len(chords)

    @staticmethod
    def _rewrite_key(key: str, mode: str) -> str:
        if mode == 'relative':
            if key.lower().endswith('min'):
                return f'{key[:-3]}maj'
            if key.lower().endswith('maj'):
                return f'{key[:-3]}min'
        return key

    def _rewrite_chords(self, chords: List[str], *, mode: str, key: str, energy: float) -> List[str]:
        base = list(chords) if chords else [key]
        if mode == 'relative':
            return [self._to_relative_chord(chord) for chord in base]
        if mode == 'cadence':
            if len(base) >= 2:
                return base[:-2] + [self._dominant_chord(key), self._tonic_chord(key)]
            return [self._dominant_chord(key), self._tonic_chord(key)]
        if mode == 'support':
            return [self._sustain_chord(chord, energy) for chord in base]
        return base

    @staticmethod
    def _to_relative_chord(chord: str) -> str:
        if 'maj7' in chord:
            return chord.replace('maj7', 'm7')
        if chord.endswith('m'):
            return chord[:-1]
        if '7' in chord:
            return chord.replace('7', 'sus4')
        return f'{chord}m'

    @staticmethod
    def _dominant_chord(key: str) -> str:
        root = key[:-3] if key.lower().endswith(('maj', 'min')) else key
        return f'{root}7'

    @staticmethod
    def _tonic_chord(key: str) -> str:
        root = key[:-3] if key.lower().endswith('maj') else key[:-3] if key.lower().endswith('min') else key
        return root if key.lower().endswith('maj') else f'{root}m'

    @staticmethod
    def _sustain_chord(chord: str, energy: float) -> str:
        if energy > 0.65:
            return f'{chord}7' if '7' not in chord else chord
        return f'{chord}sus2' if 'sus' not in chord else chord
