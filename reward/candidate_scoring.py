import importlib
import json
from pathlib import Path
from reward.objectives import AUDIO_NAMES, AUDIO_INITIAL_WEIGHTS, RewardProtocol, score_audio, aggregate

class FixedWeightCandidateScorer:

    def __init__(self, reward_inputs, config_path):
        self.evidence = json.loads(Path(reward_inputs).read_text())
        config = json.loads(Path(config_path).read_text())
        if not self.evidence.get('visual') or not self.evidence.get('measurement_source'):
            raise ValueError('refinement requires independently measured visual reward_inputs')
        if set(config.get('audio_components', AUDIO_NAMES)) != set(AUDIO_NAMES):
            raise ValueError('refinement requires all six components')
        plugin = config.get('emotion_evaluator')
        if not plugin:
            raise ValueError('refinement requires a frozen emotion evaluator')
        module, function = plugin.split(':')
        self.emotion = getattr(importlib.import_module(module), function)
        self.options = config.get('emotion_evaluator_options', {})
        self.protocol = RewardProtocol(**config.get('measurements', {}))

    def preload(self):
        if self.emotion.__module__ == 'reward.emotion' and self.emotion.__name__ == 'evaluate_musicnn':
            from reward.emotion import preload_musicnn
            preload_musicnn(**self.options)

    def __call__(self, blueprint, symbolic_outputs, audio_manifest, **kwargs):
        events = [e for item in symbolic_outputs for e in item['events']]
        duration = max((s.end_time for s in blueprint.segments))
        path = audio_manifest['soundtrack_path']
        emotion = self.emotion(audio_path=path, evidence=self.evidence, **self.options)
        result = score_audio(events, self.evidence['visual'], path, duration, self.protocol, emotion_measurement=emotion)
        result['aggregation'] = aggregate(result['components'], AUDIO_INITIAL_WEIGHTS, undefined='zero')
        result['rewards'] = {name: value['reward'] if value['status'] == 'ok' else 0.0 for name, value in result['components'].items()}
        result['rewards']['R_total'] = result['aggregation']['reward']
        result['scoring_protocol'] = 'CoMA.pdf fixed six-component acceptance; no adaptive inference weights'
        return result
