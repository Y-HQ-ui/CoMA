# CoMA Source

This source package contains the video, symbolic music, audio generation,
dataset preparation, training, reward, and evaluation implementations.

## Environment

Use Python 3.12 and the dependencies in requirements.txt. Install a PyTorch
build compatible with your hardware. Model weights and datasets are supplied
separately. No credentials, service endpoints, or machine-specific model paths
are included.

Local model discovery searches only models/ within this package. Alternatively,
set MUSICAGENT_MUPT_MODEL_DIR and MUSICAGENT_MUSICGEN_MODEL_DIR in your own
environment. The video Composer requires MUSICAGENT_BASE_URL,
MUSICAGENT_API_KEY, and MUSICAGENT_MODEL_NAME. These have no working defaults.

## Local Verification

Run from the package root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m scripts.verify_dataset --output-dir output/dataset_check
```

The dataset check creates synthetic MIDI, audio, and video fixtures solely for
software verification. It does not reproduce research results.

## Dataset Preparation

The dataset guide is docs/DATASET_PREPARATION_ZH.md. Input paths below are
relative examples that must be populated with your own authorized data.

```bash
python -m training.prepare_dataset slakh --source datasets/slakh2100 --output-dir data/slakh
python -m training.prepare_dataset musicpro --index datasets/musicpro/index.jsonl --output-dir data/musicpro --extract-visual
python -m training.prepare_dataset cineparse --index datasets/visual/annotations.jsonl --output-dir data/visual --exclude-manifest data/musicpro/all.jsonl
python -m training.prepare_dataset mix-audio --slakh-manifest data/slakh/audio_train.jsonl --musicpro-manifest data/musicpro/audio_train.jsonl --output-dir data/audio_mixture
```

## Generation And Training

```bash
python run_coma_pipeline.py --blueprint-json examples/blueprint_smoke.json --output-dir output/example --generate-symbolic --generate-audio --mupt-device cpu --musicgen-device cpu
python -m training.run_staged --config configs/staged_training.example.json --output-dir output/training
```
