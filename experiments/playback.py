from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import subprocess
from experiments.provenance import sha256_file
from metrics.waveform import prepare_evaluation_mix

def _run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} failed: {result.stderr[-1800:]}')
    return result

def streams(path):
    return json.loads(_run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]).stdout)['streams']

def measure(path):
    result = _run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-map', '0:a:0', '-af', 'loudnorm=I=-18:TP=-1:LRA=11:print_format=json', '-f', 'null', '-'])
    start = result.stderr.rfind('{')
    if start < 0:
        raise ValueError('FFmpeg did not return loudness measurements')
    values = json.JSONDecoder().raw_decode(result.stderr[start:])[0]
    loudness, peak = (float(values['input_i']), float(values['input_tp']))
    if math.isnan(loudness) or math.isnan(peak) or loudness == math.inf or (peak == math.inf):
        raise ValueError('invalid loudness measurement')
    return {'integrated_lufs': loudness if math.isfinite(loudness) else None, 'true_peak_dbtp': peak if math.isfinite(peak) else None}

def constant_gain(measurement, target_lufs=-18.0, peak_limit_dbtp=-1.0):
    loudness, peak = (measurement['integrated_lufs'], measurement['true_peak_dbtp'])
    if loudness is None:
        return min(0.0, peak_limit_dbtp - peak) if peak is not None else 0.0
    if peak is None:
        raise ValueError('finite loudness requires a measured true peak')
    return min(target_lufs - loudness, peak_limit_dbtp - peak - 0.0001)

def prepare_audio(source, destination):
    source, destination = (Path(source).resolve(), Path(destination).resolve())
    if source == destination or destination.exists():
        raise FileExistsError('playback copy must be a new file')
    original_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='coma-playback-') as tmp:
        canonical = Path(tmp) / 'objective.wav'
        mix = prepare_evaluation_mix(source, canonical)
        before = measure(canonical)
        gain = constant_gain(before)

        def render():
            _run(['ffmpeg', '-v', 'error', '-i', str(canonical), '-map', '0:a:0', '-af', f'volume={gain:.10f}dB:precision=double', '-c:a', 'pcm_f32le', '-map_metadata', '-1', '-y', str(destination)])
        render()
        after = measure(destination)
        if after['true_peak_dbtp'] is not None and after['true_peak_dbtp'] > -1.0:
            gain -= after['true_peak_dbtp'] + 1.0 + 0.01
            render()
            after = measure(destination)
        if after['true_peak_dbtp'] is not None and after['true_peak_dbtp'] > -1.0:
            raise ValueError('playback copy exceeds the true-peak limit')
        if sha256_file(source) != original_hash:
            raise ValueError('objective source changed during playback preparation')
        return {'kind': 'constant_gain_playback', 'source_sha256': original_hash, 'evaluation_mix': {k: v for k, v in mix.items() if k != 'path'}, 'gain_db': gain, 'target_lufs': -18.0, 'peak_limit_dbtp': -1.0, 'before': before, 'after': after, 'loudness_status': 'measured' if before['integrated_lufs'] is not None else 'below_gate_or_silent', 'sha256': sha256_file(destination)}

def prepare_video(source, destination):
    source, destination = (Path(source).resolve(), Path(destination).resolve())
    if source == destination or destination.exists():
        raise FileExistsError('silent video must be a new file')
    original_hash = sha256_file(source)
    if not any((s['codec_type'] == 'video' for s in streams(source))):
        raise ValueError('listening stimulus has no video stream')
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(['ffmpeg', '-v', 'error', '-i', str(source), '-map', '0:v:0', '-an', '-c:v', 'copy', '-map_metadata', '-1', '-map_chapters', '-1', '-y', str(destination)])
    kinds = [s['codec_type'] for s in streams(destination)]
    if kinds != ['video'] or sha256_file(source) != original_hash:
        raise ValueError('silent video verification failed')
    return {'kind': 'muted_video', 'source_sha256': original_hash, 'audio_streams': 0, 'sha256': sha256_file(destination)}
