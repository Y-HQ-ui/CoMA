import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.io import wavfile

def verify(spec, output_dir, mmp_protocol=None):
    from .preflight import check
    from . import frozen_backends as backends
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    t = np.arange(480000) / 32000
    wave = (0.05 + 0.03 * np.sin(2 * np.pi * 0.4 * t)) * np.sin(2 * np.pi * (220 * t + 3 * t * t))
    audio = root / 'synthetic_probe.wav'
    wavfile.write(audio, 32000, wave.astype(np.float32))
    reports = check(spec)['backends']
    for name, options in spec.get('backends', {}).items():
        if name == 'vggish':
            reports[name]['missing'] = [v for v in reports[name]['missing'] if v != 'local_path:reference_audio']
        if reports[name]['missing']:
            reports[name]['status'] = 'unavailable'
            continue
        try:
            if name == 'emotion':
                values, times, protocol = backends.emotion_features(audio, options)
                if values.shape != (25, 2) or len(times) != 25:
                    raise ValueError('unexpected DEAM windows/axes')
            elif name == 'genre':
                values, protocol = backends.genre_probabilities(audio, options)
                if not np.allclose(values.sum(1), 1.0):
                    raise ValueError('non-normalized genre probabilities')
            elif name == 'clap':
                values, text, protocol = backends.clap_embeddings(audio, options, 'instrumental music')
                if values.shape[1:] != text.shape:
                    raise ValueError('CLAP audio/text dimensions differ')
            elif name == 'vggish':
                values, protocol = backends.vggish_embeddings(audio, options)
            elif name == 'imagebind':
                video = root / 'synthetic_video.mp4'
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=256x256:rate=10', '-t', '15', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(video)], check=True, capture_output=True, timeout=120)
                left, right, protocol = backends.imagebind_embeddings(audio, video, options)
                if left.shape != right.shape:
                    raise ValueError('ImageBind audio/video dimensions differ')
                values = np.stack([left, right])
            else:
                raise ValueError('unknown backend')
            if not values.size or not np.isfinite(values).all():
                raise ValueError('empty/nonfinite backend outputs')
            reports[name] = {'status': 'executed', 'shape': list(values.shape), 'protocol': protocol}
        except Exception as exc:
            reports[name] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    if mmp_protocol is not None:
        from .mmp_backends import transcribe_mt3
        try:
            result = transcribe_mt3(audio_path=audio, checkpoint=mmp_protocol['transcriber_checkpoint'], decoding=mmp_protocol['decoding'])
            reports['mt3'] = {'status': 'executed', 'transcription': result}
        except Exception as exc:
            reports['mt3'] = {'status': 'unavailable_or_failed', 'error': f'{type(exc).__name__}: {exc}'}
    result = {'scope': 'synthetic implementation probe; not experimental quality scores', 'llm_called': False, 'backends': reports, 'all_executed': bool(reports) and all((r['status'] == 'executed' for r in reports.values()))}
    (root / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mmp-protocol')
    args = p.parse_args(argv)
    from .evaluate import resolve_metric_paths
    source = Path(args.config).resolve()
    spec = resolve_metric_paths(json.loads(source.read_text()), source.parent)
    protocol = None
    if args.mmp_protocol:
        from .uniform_transcription import resolve_protocol_paths
        path = Path(args.mmp_protocol).resolve()
        protocol = resolve_protocol_paths(json.loads(path.read_text()), path.parent)
    result = verify(spec, args.output_dir, protocol)
    print(json.dumps(result, indent=2))
    return int(not result['all_executed'])
if __name__ == '__main__':
    raise SystemExit(main())
