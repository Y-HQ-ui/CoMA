from functools import lru_cache
import importlib
import math
from pathlib import Path
import sys
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from experiments.provenance import sha256_file

def audio_at(path, sample_rate):
    wave, sr = sf.read(path, always_2d=True, dtype='float32')
    wave = wave.mean(1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('finite nonempty audio required')
    if sr != sample_rate:
        divisor = math.gcd(sr, sample_rate)
        wave = resample_poly(wave, sample_rate // divisor, sr // divisor)
    return wave.astype(np.float32)

def local_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)

def graph_binding(options):
    paths = [local_file(options[key]) for key in ('embedding_path', 'head_path')]
    return (paths, {p: sha256_file(p) for p in paths})

@lru_cache(maxsize=4)
def musicnn_graphs(embedding_path, head_path, embedding_hash, head_hash, output):
    from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
    return (TensorflowPredictMusiCNN(graphFilename=embedding_path, output='model/dense/BiasAdd'), TensorflowPredict2D(graphFilename=head_path, output=output))

def emotion_features(audio, options):
    if options.get('model_id') != 'deam-msd-musicnn-2':
        raise ValueError('EAS requires deam-msd-musicnn-2')
    axes = options.get('output_axes')
    if sorted(axes or []) != ['arousal', 'valence']:
        raise ValueError('declare graph output_axes from its metadata')
    paths, hashes = graph_binding(options)
    embedding, head = musicnn_graphs(*paths, *[hashes[p] for p in paths], options.get('output', 'model/Identity'))
    wave = audio_at(audio, 16000)
    starts = list(range(0, len(wave) - 48000 + 1, 8000))
    values = []
    for start in starts:
        prediction = np.asarray(head(embedding(wave[start:start + 48000])), float)
        if prediction.size == 0 or prediction.shape[-1] != 2 or (not np.isfinite(prediction).all()):
            raise ValueError('emotion graph must return finite two-axis predictions')
        values.append(prediction.reshape(-1, 2).mean(0)[[axes.index('valence'), axes.index('arousal')]])
    if not values:
        raise ValueError('no complete 3-second emotion windows')
    return (np.asarray(values), np.asarray(starts) / 16000 + 1.5, {'checkpoint': 'deam-msd-musicnn-2', 'checkpoint_sha256': hashes, 'sample_rate': 16000, 'window_seconds': 3.0, 'hop_seconds': 0.5, 'axis_order': ['valence', 'arousal'], 'raw_output_axes': axes})

def genre_probabilities(audio, options):
    if options.get('model_id') != 'genre_tzanetakis-musicnn-msd-1':
        raise ValueError('IS requires the ten-genre MusicNN classifier')
    if not options.get('logits_output'):
        raise ValueError('declare logits_output from the graph metadata; probabilities are not logits')
    paths, hashes = graph_binding(options)
    embedding, head = musicnn_graphs(*paths, *[hashes[p] for p in paths], options['logits_output'])
    logits = np.asarray(head(embedding(audio_at(audio, 16000))), float)
    if logits.ndim != 2 or logits.shape[1] != 10 or (not len(logits)) or (not np.isfinite(logits).all()):
        raise ValueError('genre classifier must emit finite [patches,10] logits')
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    return (probabilities, {'checkpoint': options['model_id'], 'checkpoint_sha256': hashes, 'logits_output': options['logits_output'], 'softmax_temperature': 1.0, 'clip_pooling': 'mean_patch_probabilities'})

@lru_cache(maxsize=2)
def clap_models(model_dir, device):
    import torch
    from transformers import AutoProcessor, ClapModel
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = ClapModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    if processor.feature_extractor.sampling_rate != 48000:
        raise ValueError('CLAP frontend must use 48 kHz')
    return (processor, model)

def clap_embeddings(audio, options, prompt=''):
    import torch
    if options.get('model_id') != 'laion/larger_clap_music_and_speech':
        raise ValueError('conditional metrics require laion/larger_clap_music_and_speech')
    directory = str(Path(options['model_dir']).resolve())
    device = options.get('device', 'cpu')
    processor, model = clap_models(directory, device)
    seconds = float(options.get('segment_seconds', 10.0))
    if not 0 < seconds <= 10:
        raise ValueError('CLAP segments must be in (0,10] seconds to avoid random long-clip truncation')
    wave = audio_at(audio, 48000)
    size = round(seconds * 48000)
    segments = [wave[start:start + size] for start in range(0, len(wave), size)]
    embeddings = []
    with torch.inference_mode():
        for segment in segments:
            inputs = processor(audios=[segment], sampling_rate=48000, return_tensors='pt', padding='repeatpad')
            features = model.get_audio_features(**{key: value.to(device) for key, value in inputs.items()})
            embeddings.append(features.detach().cpu().numpy()[0])
        text_embedding = None
        if prompt:
            inputs = processor(text=[prompt], return_tensors='pt', padding=True, truncation=True)
            text_embedding = model.get_text_features(**{key: value.to(device) for key, value in inputs.items()}).detach().cpu().numpy()[0]
    from experiments.provenance import checkpoint_fingerprint
    return (np.asarray(embeddings), text_embedding, {'checkpoint': options['model_id'], 'checkpoint_fingerprint': checkpoint_fingerprint(directory), 'sample_rate': 48000, 'segment_seconds': seconds, 'last_segment': 'retained; deterministic repeatpad', 'pooling': 'mean_segment_embeddings_then_cosine'})

def official_module(source_root, module):
    root = str(Path(source_root).resolve())
    if not Path(root).is_dir():
        raise FileNotFoundError(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    loaded = importlib.import_module(module)
    if not Path(loaded.__file__).resolve().is_relative_to(root):
        raise ValueError(f'{module} was imported from a different checkout')
    return loaded

def vggish_embeddings(audio, options):
    if options.get('model_id') != 'vggish':
        raise ValueError('FAD requires VGGish embeddings')
    import tensorflow.compat.v1 as tf
    source = options['source_root']
    frontend = official_module(source, 'vggish_input')
    network = official_module(source, 'vggish_slim')
    params = official_module(source, 'vggish_params')
    checkpoint = local_file(options['checkpoint'])
    if not np.isclose(params.EXAMPLE_WINDOW_SECONDS, 0.96) or not np.isclose(params.EXAMPLE_HOP_SECONDS, 0.96):
        raise ValueError('VGGish frontend must use .96-second nonoverlapping examples')
    wave = audio_at(audio, 16000)
    examples = frontend.waveform_to_examples(wave, 16000)[:len(wave) // 15360]
    if not len(examples):
        raise ValueError('no complete VGGish examples')
    graph = tf.Graph()
    with graph.as_default():
        network.define_vggish_slim(training=False)
        with tf.Session(graph=graph) as session:
            network.load_vggish_slim_checkpoint(session, checkpoint)
            values = session.run(graph.get_tensor_by_name(params.OUTPUT_TENSOR_NAME), feed_dict={graph.get_tensor_by_name(params.INPUT_TENSOR_NAME): examples})
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != 128 or (not np.isfinite(values).all()):
        raise ValueError('VGGish must emit finite 128-D embeddings')
    return (values, {'checkpoint': 'vggish', 'checkpoint_sha256': sha256_file(checkpoint), 'frontend_sha256': sha256_file(frontend.__file__), 'network_sha256': sha256_file(network.__file__), 'sample_rate': 16000, 'window_seconds': 0.96, 'hop_seconds': 0.96, 'pca': False, 'quantization': False})

def normalized_segment_mean(segments):
    values = np.asarray(segments, float)
    if values.ndim != 2 or not len(values) or (not np.isfinite(values).all()):
        raise ValueError('finite segment embeddings required')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError('zero segment embedding')
    mean = (values / norms).mean(0)
    norm = np.linalg.norm(mean)
    if norm <= 1e-12:
        raise ValueError('zero pooled embedding')
    return mean / norm

def imagebind_embeddings(audio, video, options):
    import tempfile
    import subprocess
    import torch
    if options.get('model_id') != 'imagebind_huge':
        raise ValueError('ImageBind requires imagebind_huge')
    source = options['source_root']
    data = official_module(source, 'imagebind.data')
    module = official_module(source, 'imagebind.models.imagebind_model')
    checkpoint = local_file(options['checkpoint'])
    device = options.get('device', 'cpu')
    model = module.imagebind_huge(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.requires_grad_(False)
    starts = np.linspace(0.0, 13.0, 5)
    wave = audio_at(audio, 16000)
    if len(wave) != 240000:
        raise ValueError('ImageBind expects the canonical 15-second evaluation mix')
    audio_embeddings = []
    video_embeddings = []
    with tempfile.TemporaryDirectory(prefix='coma-imagebind-') as tmp, torch.inference_mode():
        for i, start in enumerate(starts):
            clip_audio = Path(tmp) / f'{i}.wav'
            clip_video = Path(tmp) / f'{i}.mp4'
            sf.write(clip_audio, wave[round(start * 16000):round((start + 2) * 16000)], 16000, subtype='FLOAT')
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-ss', str(start), '-t', '2', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(clip_video)], check=True, capture_output=True)
            inputs = {module.ModalityType.AUDIO: data.load_and_transform_audio_data([str(clip_audio)], device, sample_rate=16000, clip_duration=2, clips_per_video=1), module.ModalityType.VISION: data.load_and_transform_video_data([str(clip_video)], device, clip_duration=2, clips_per_video=1, sample_rate=16000)}
            values = model(inputs)
            audio_embeddings.append(values[module.ModalityType.AUDIO].detach().cpu().numpy()[0])
            video_embeddings.append(values[module.ModalityType.VISION].detach().cpu().numpy()[0])
    return (normalized_segment_mean(audio_embeddings), normalized_segment_mean(video_embeddings), {'checkpoint': 'imagebind_huge', 'checkpoint_sha256': sha256_file(checkpoint), 'source_sha256': sha256_file(module.__file__), 'transforms_sha256': sha256_file(data.__file__), 'video_sha256': sha256_file(video), 'segments': 5, 'seconds_per_segment': 2, 'frames_per_segment': 2, 'start_seconds': starts.tolist(), 'pooling': 'L2_each_segment; mean; L2_again', 'similarity': 'raw_cosine'})

def materialize(audio, directory, configurations, *, prompt='', reference='', emotion_target=None, video=''):
    from .protocol import imagebind_score
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    updates = {}
    overrides = {}
    reports = {}
    distribution = {}

    def array(name, values):
        path = directory / (name + '.npy')
        np.save(path, np.asarray(values))
        return str(path)
    for name, options in configurations.items():
        try:
            if name == 'emotion':
                values, times, protocol = emotion_features(audio, options)
                target = emotion_target or options.get('target')
                if isinstance(target, str):
                    import json
                    target = json.loads(Path(target).read_text())
                if not target or target.get('used_target_audio') is not False or (not target.get('source')):
                    raise ValueError('EAS needs independently observed silent-video VA annotations')
                source_times = np.asarray(target['times'], float)
                source_values = np.asarray(target['values'], float)
                if source_times.ndim != 1 or len(source_times) < 2 or (np.diff(source_times) <= 0).any() or (source_values.shape != (len(source_times), 2)):
                    raise ValueError('emotion target must have ordered times and VA pairs')
                if not np.isfinite(source_times).all() or not np.isfinite(source_values).all():
                    raise ValueError('nonfinite emotion annotations')
                mask = (times >= source_times[0]) & (times <= source_times[-1])
                if mask.sum() < 20:
                    raise ValueError('EAS needs at least 20 common valid points')
                target_values = np.column_stack([np.interp(times[mask], source_times, source_values[:, i]) for i in range(2)])
                updates['emotion'] = {'generated': array('emotion_audio', values[mask]), 'reference': array('emotion_video', target_values), 'checkpoint': protocol['checkpoint'], 'protocol': protocol, 'sample_hz': 2.0, 'max_lag_seconds': 1.0, 'times': times[mask].tolist(), 'target_source': target['source']}
            elif name == 'clap':
                values, text, protocol = clap_embeddings(audio, options, prompt)
                if reference:
                    ref, _, _ = clap_embeddings(reference, options)
                    updates['style'] = {'generated': array('clap_generated', values), 'reference': array('clap_reference', ref), 'checkpoint': protocol['checkpoint'], 'protocol': protocol, 'reference_audio_sha256': sha256_file(reference)}
                if text is not None:
                    updates['prompt_alignment_measurement'] = {'value': (imagebind_score(text, values.mean(0)) + 1) / 2, 'protocol': protocol}
            elif name == 'genre':
                probabilities, protocol = genre_probabilities(audio, options)
                distribution['inception_score'] = {'status': 'ok', 'backend': 'frozen_musicnn_genre_logits', 'probabilities_path': array('genre_probabilities', probabilities), 'protocol': protocol, 'level': 'distribution_features; aggregate over common valid clips'}
            elif name == 'vggish':
                values, protocol = vggish_embeddings(audio, options)
                if not options.get('reference_audio') or options.get('reference_role') != 'held_out_evaluation':
                    raise ValueError('FAD needs a held-out target for paired common-valid-set aggregation')
                from .waveform import prepare_evaluation_mix
                reference_mix = prepare_evaluation_mix(options['reference_audio'], directory / 'fad_reference.wav')
                ref, reference_protocol = vggish_embeddings(reference_mix['path'], options)
                distribution['frechet_audio_distance'] = {'status': 'ok', 'backend': 'frozen_vggish_raw_128', 'embeddings_path': array('vggish_generated', values), 'reference_embeddings_path': array('vggish_reference', ref), 'protocol': protocol, 'reference_audio_sha256': reference_mix['audio_sha256'], 'level': 'distribution_features; recompute per common seed set'}
            elif name == 'imagebind':
                audio_embedding, video_embedding, protocol = imagebind_embeddings(audio, video or options['video'], options)
                overrides.update(imagebind_audio_embedding=array('imagebind_audio', audio_embedding), imagebind_video_embedding=array('imagebind_video', video_embedding))
                updates['imagebind_checkpoint'] = protocol['checkpoint']
                updates['imagebind_protocol'] = protocol
            else:
                raise ValueError(f'unknown frozen metric backend: {name}')
            reports[name] = {'status': 'ok', 'protocol': protocol}
        except (ImportError, OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            reports[name] = {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'}
    return (updates, overrides, reports, distribution)
