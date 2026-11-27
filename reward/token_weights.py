from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import os
import numpy as np
SYSTEM_PROMPT = 'You control CoMA training reward coefficients. Return JSON only with a logits object containing exactly the supplied component names and finite real values, and a rationale. Use measured component summaries and optional human feedback. Weight changes are applied externally through temperature softmax, a simplex floor, smoothing and a Euclidean step bound. Do not report quality measurements or invent unavailable evidence.'

@dataclass(frozen=True)
class WeightConfig:
    mode: str = 'fixed'
    temperature: float = 1.0
    rho: float = 0.1
    floor: float = 0.0001
    max_step: float = 0.1
    update_every: int = 8
    consistency_final_gain: float = 1.0
    on_error: str = 'hold'

    def __post_init__(self):
        if self.mode not in {'fixed', 'schedule', 'rule', 'llm'} or self.on_error not in {'hold', 'error'}:
            raise ValueError('invalid controller mode/error behavior')
        if not np.isfinite([self.temperature, self.rho, self.floor, self.max_step, self.consistency_final_gain]).all():
            raise ValueError('nonfinite weight configuration')
        if self.temperature <= 0 or not 0 <= self.rho <= 1 or self.floor < 0 or (self.max_step < 0) or (self.update_every < 1) or (self.consistency_final_gain < 1):
            raise ValueError('invalid coefficient constraint')
        if self.mode == 'fixed' and self.consistency_final_gain != 1:
            raise ValueError('fixed weights cannot have a ramp; choose explicit schedule mode')

def map_logits(logits, temperature, floor):
    values = np.asarray(logits, np.float64)
    if values.ndim != 1 or not len(values) or (not np.isfinite(values).all()) or (len(values) * floor > 1):
        raise ValueError('invalid logits or infeasible simplex floor')
    scaled = (values - values.max()) / temperature
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    return floor + (1 - len(values) * floor) * probabilities

class WeightController:

    def __init__(self, names, config=None, *, initial=None, llm=None):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('nonempty unique component names required')
        self.config = config or WeightConfig()
        if len(self.names) * self.config.floor > 1:
            raise ValueError('simplex floor infeasible for active components')
        initial = initial or {name: 1 / len(self.names) for name in self.names}
        values = np.asarray([initial[name] for name in self.names], float)
        if set(initial) != set(self.names) or (values < self.config.floor).any() or (not np.isfinite(values).all()) or (not np.isclose(values.sum(), 1)):
            raise ValueError('initial coefficients must satisfy the declared simplex and floor')
        self.initial, self.weights = (values.copy(), values.copy())
        self.llm = llm
        self.pending, self.history = ([], [])

    def update(self, step, rewards, *, progress=0.0, human_feedback=None):
        if set(rewards) != set(self.names) or not all((np.isfinite(v) and 0 <= v <= 1 for v in rewards.values())):
            raise ValueError('controller requires bounded observations for exactly its active objectives')
        if not 0 <= progress <= 1:
            raise ValueError('invalid training progress')
        self.pending.append([rewards[name] for name in self.names])
        before = self.weights.copy()
        record = {'step': step, 'mode': self.config.mode, 'normalization': 'single_global_simplex', 'before': dict(zip(self.names, before.tolist())), 'progress': progress, 'updated': False}
        if self.config.mode != 'fixed' and (step + 1) % self.config.update_every == 0:
            summaries = dict(zip(self.names, np.mean(self.pending, axis=0).tolist()))
            self.pending = []
            record['reward_summary'] = summaries
            try:
                if self.config.mode == 'llm':
                    if self.llm is None:
                        raise ValueError('LLM weight mode requires configured endpoint/callback')
                    response = self.llm({'components': list(self.names), 'reward_summary': summaries, 'previous_weights': record['before'], 'human_feedback': human_feedback})
                    record['response'] = response
                    if not isinstance(response.get('logits'), dict) or set(response['logits']) != set(self.names):
                        raise ValueError('LLM returned wrong coefficient names')
                    logits = np.asarray([response['logits'][name] for name in self.names], float)
                elif self.config.mode == 'rule':
                    logits = np.array([1 - summaries[name] for name in self.names])
                else:
                    logits = np.log(self.initial)
                if 'consistency' in self.names:
                    logits[self.names.index('consistency')] += np.log(1 + (self.config.consistency_final_gain - 1) * progress)
                proposal = map_logits(logits, self.config.temperature, self.config.floor)
                delta = self.config.rho * (proposal - before)
                norm = np.linalg.norm(delta)
                if norm > self.config.max_step:
                    delta *= self.config.max_step / norm
                self.weights = before + delta
                record.update(updated=True, logits=logits.tolist(), proposal=proposal.tolist())
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                if self.config.on_error == 'error':
                    raise
                record.update(error=str(exc), fallback='hold_previous_coefficients')
        if self.config.mode == 'fixed':
            self.pending = []
        record.update(weights=dict(zip(self.names, self.weights.tolist())), sum=float(self.weights.sum()), l2_step=float(np.linalg.norm(self.weights - before)))
        self.history.append(record)
        return (record['weights'], record)

    def state_dict(self):
        return {'names': list(self.names), 'config': asdict(self.config), 'initial': self.initial.tolist(), 'weights': self.weights.tolist(), 'pending': self.pending}

    def load_state_dict(self, state):
        if state['names'] != list(self.names) or state['config'] != asdict(self.config):
            raise ValueError('controller checkpoint configuration mismatch')
        values = np.asarray(state['weights'], float)
        if values.shape != self.weights.shape or not np.isfinite(values).all() or (values < self.config.floor - 1e-12).any() or (not np.isclose(values.sum(), 1)):
            raise ValueError('invalid saved controller coefficients')
        self.weights = values
        self.initial = np.asarray(state['initial'], float)
        self.pending = state['pending']

class HTTPWeightController:

    def __init__(self, base_url=None, model=None, api_key=None, timeout=120.0):
        from runtime_protocol import DEFAULT_LLM_MODEL
        self.base_url = base_url or os.getenv('COMA_WEIGHT_BASE_URL') or os.getenv('MUSICAGENT_BASE_URL', '')
        self.model = model or os.getenv('COMA_WEIGHT_MODEL') or os.getenv('MUSICAGENT_MODEL_NAME') or DEFAULT_LLM_MODEL
        self.api_key = api_key if api_key is not None else os.getenv('COMA_WEIGHT_API_KEY') or os.getenv('MUSICAGENT_API_KEY', '')
        self.timeout = timeout
        if not self.base_url or not self.model:
            raise ValueError('set MUSICAGENT_BASE_URL or COMA_WEIGHT_BASE_URL')
        self.traces = []

    def __call__(self, payload):
        import requests
        body = {'model': self.model, 'temperature': 0.0, 'max_tokens': 512, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': json.dumps(payload, allow_nan=False)}]}
        trace = {'request': body, 'timeout': self.timeout}
        try:
            response = requests.post(self.base_url.rstrip('/') + '/chat/completions', json=body, headers={'Authorization': f'Bearer {self.api_key}'}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            trace.update(response=content, served_model=data.get('model'), usage=data.get('usage'))
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError('weight response must be a JSON object')
            return result
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            trace['error_type'] = type(exc).__name__
            raise RuntimeError(f'weight endpoint failed: {type(exc).__name__}') from exc
        finally:
            self.traces.append(trace)
