import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from experiments.provenance import sha256_file
from runtime_protocol import DEFAULT_LLM_MODEL
AXES = ('mode', 'melody', 'harmony', 'rhythm', 'expression')
DEFAULT_PROMPT = 'Evaluate only the anonymous transcribed ABC below. Rate mode, melody, harmony, rhythm, and expression using integers from 1 (poor) to 5 (excellent). Return exactly a JSON object with these five keys. Do not infer a generating system or compare to hidden reference music.'
SCORE_DEFINITION = '(mean of five integer ratings in [1,5] - 1) / 4'

def midi_to_abc(path):
    from training.prepare_dataset import midi_events, abc_voices
    events, _ = midi_events(path)
    duration = max((e.onset + e.duration for e in events))
    groups = {}
    for event in events:
        groups.setdefault((event.track_id, event.program), []).append(event)
    lines = ['X:1', 'T:Anonymous transcription', 'M:4/4', 'L:1/8', 'Q:1/4=120', 'K:C']
    count = 0
    for (_, program), track in sorted(groups.items()):
        for voice in abc_voices(track, bpm=120, duration=duration):
            count += 1
            lines.extend([f'V:{count}', f'%%MIDI program {program}', voice.rstrip()])
    if not count:
        raise ValueError('MT3 MIDI contains no sounding voices')
    return '\n'.join(lines) + '\n'

def transcribe_mt3(*, audio_path, checkpoint, decoding):
    command = decoding.get('mt3_command')
    runner = Path(__file__).with_name('mt3_runner.py')
    built_in = command is None
    if built_in:
        source = decoding.get('mt3_source_root')
        if not source or not Path(source).is_dir():
            raise ValueError('decoding.mt3_source_root must point to the local official MT3 checkout')
        command = [decoding.get('mt3_python') or sys.executable, str(runner), '--source-root', str(Path(source).resolve()), '--audio', '{audio}', '--checkpoint', '{checkpoint}', '--midi', '{midi}']
    else:
        runner = Path(decoding.get('mt3_source_file', ''))
    if not isinstance(command, list) or not command or any((not isinstance(arg, str) for arg in command)):
        raise ValueError('decoding.mt3_command must be an official local MT3 argv template with {audio}, {midi}, {checkpoint}')
    for placeholder in ('{audio}', '{midi}', '{checkpoint}'):
        if not any((placeholder in arg for arg in command)):
            raise ValueError(f'missing MT3 command placeholder {placeholder}')
    if not runner.is_file():
        raise ValueError('record mt3_source_file for the local official inference runner')
    checkpoint_path = Path(checkpoint).resolve()
    files = sorted((p for p in checkpoint_path.rglob('*') if p.is_file())) if checkpoint_path.is_dir() else [checkpoint_path]
    if not files or any((not p.is_file() for p in files)):
        raise FileNotFoundError(checkpoint_path)
    binding = {str(p): sha256_file(p) for p in files}
    with TemporaryDirectory(prefix='coma-mt3-') as tmp:
        midi = Path(tmp) / 'transcription.mid'
        values = {'audio': str(Path(audio_path).resolve()), 'midi': str(midi), 'checkpoint': str(checkpoint_path)}
        argv = [arg.format(**values) for arg in command]
        try:
            subprocess.run(argv, check=True, capture_output=True, text=True, timeout=float(decoding.get('mt3_timeout', 600)))
        except subprocess.SubprocessError as exc:
            detail = getattr(exc, 'stderr', '') or ''
            if isinstance(detail, bytes):
                detail = detail.decode(errors='replace')
            raise RuntimeError(f'MT3 process failed: {type(exc).__name__}: {detail[-2000:]}') from exc
        if not midi.is_file():
            raise ValueError('MT3 did not produce MIDI')
        abc = midi_to_abc(midi)
        return {'abc': abc, 'checkpoint': checkpoint, 'checkpoint_sha256': binding, 'audio_sha256': sha256_file(audio_path), 'midi_sha256': sha256_file(midi), 'runner_sha256': sha256_file(runner), 'implementation': 'bundled_official_notebook_adapter' if built_in else 'explicit_external_command', 'inference_sha256': sha256_file(Path(__file__).with_name('mt3_official.py')) if built_in else None, 'official_source_sha256': {str(p.relative_to(source)): sha256_file(p) for p in sorted((Path(source) / 'mt3').rglob('*')) if p.is_file() and p.suffix in {'.py', '.gin'}} if built_in else None, 'conversion': 'MIDI event times; deterministic multivoice ABC at fixed 120 BPM; no target/native-score input'}

def evaluate_mmp(*, abc, checkpoint, prompt, decoding, score_definition):
    import requests
    if checkpoint != DEFAULT_LLM_MODEL:
        raise ValueError(f'MMP evaluator must be {DEFAULT_LLM_MODEL}')
    if decoding.get('temperature') != 0 or decoding.get('max_tokens') != 2048:
        raise ValueError('MMP requires temperature=0 and max_tokens=2048')
    if score_definition != SCORE_DEFINITION:
        raise ValueError('unexpected MMP score definition')
    base = os.getenv('COMA_MMP_BASE_URL') or os.getenv('MUSICAGENT_BASE_URL', '')
    key = os.getenv('COMA_MMP_API_KEY') or os.getenv('MUSICAGENT_API_KEY', '')
    if not base or not key:
        raise RuntimeError('configure the MMP API endpoint and key')
    body = {'model': checkpoint, 'temperature': 0, 'max_tokens': 2048, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': abc}]}
    try:
        response = requests.post(base.rstrip('/') + '/chat/completions', json=body, headers={'Authorization': f'Bearer {key}'}, timeout=120)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f'MMP evaluator request failed: {type(exc).__name__}') from exc
    if data.get('model') != checkpoint:
        raise ValueError('MMP served model differs from the requested frozen evaluator')
    ratings = json.loads(data['choices'][0]['message']['content'])
    if not isinstance(ratings, dict) or set(ratings) != set(AXES) or any((type(value) is not int or not 1 <= value <= 5 for value in ratings.values())):
        raise ValueError('MMP evaluator must return exactly five integer ratings in [1,5]')
    return {'score': (sum(ratings.values()) / 5 - 1) / 4, 'ratings': ratings, 'checkpoint': checkpoint, 'requested_model': checkpoint, 'served_model': data['model'], 'temperature': 0, 'max_tokens': 2048, 'score_definition': score_definition}
