import numpy as np

def assemble_timeline(waveforms, intervals, sample_rate, *, duration=None, fade_seconds=0.01):
    if len(waveforms) != len(intervals) or not waveforms or sample_rate <= 0:
        raise ValueError('waveforms/intervals must be nonempty and aligned')
    duration = duration if duration is not None else max((end for _, end in intervals))
    total = int(round(duration * sample_rate))
    output = np.zeros(total, dtype=np.float64)
    weights = np.zeros(total, dtype=np.float64)
    for audio, (start, end) in zip(waveforms, intervals):
        left, right = (round(start * sample_rate), round(end * sample_rate))
        if left < 0 or right <= left or right > total:
            raise ValueError('invalid render interval')
        audio = np.asarray(audio, dtype=np.float64).reshape(-1)
        if audio.size < right - left:
            raise ValueError('rendered audio is shorter than requested interval')
        audio = audio[:right - left].copy()
        fade = min(round(fade_seconds * sample_rate), len(audio) // 2)
        if fade:
            ramp = np.linspace(0, 1, fade)
            audio[:fade] *= ramp
            audio[-fade:] *= ramp[::-1]
        output[left:right] += audio
        weights[left:right] += 1
    if np.any(weights == 0):
        raise ValueError('render intervals leave uncovered samples')
    return (output / weights).astype(np.float32)

def replace_audio_interval(original, candidate, *, start_sample, end_sample, fade_samples=320):
    original = np.asarray(original)
    candidate = np.asarray(candidate)
    if original.shape != candidate.shape or not 0 <= start_sample < end_sample <= len(original):
        raise ValueError('local replacement needs equal shapes and a valid mask')
    result = original.copy()
    n = end_sample - start_sample
    fade = min(fade_samples, n // 2)
    weights = np.ones(n)
    if fade:
        weights[:fade] = np.linspace(0, 1, fade)
        weights[-fade:] = np.linspace(1, 0, fade)
    if original.ndim == 2:
        weights = weights[:, None]
    result[start_sample:end_sample] = original[start_sample:end_sample] * (1 - weights) + candidate[start_sample:end_sample] * weights
    return result
