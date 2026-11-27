import numpy as np

def simplex_with_floor(scores, floor=0.0001, temperature=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or (not np.isfinite(scores).all()) or (scores < 0).any():
        raise ValueError('weight proposals must be finite, nonnegative vectors')
    if not 0 <= floor < 1 / len(scores) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('infeasible floor or invalid softmax temperature')
    logits = np.log(np.maximum(scores, np.finfo(np.float64).tiny)) / temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return floor + (1 - len(scores) * floor) * probs

def bounded_update(previous, proposal, *, smoothing, max_step):
    previous = np.asarray(previous, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if not 0 <= smoothing <= 1 or max_step < 0 or (not np.isfinite(max_step)):
        raise ValueError('invalid smoothing/step bound')
    delta = smoothing * (proposal - previous)
    norm = np.linalg.norm(delta)
    if norm > max_step:
        delta *= max_step / norm
    return previous + delta
