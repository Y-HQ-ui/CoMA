"""Check the tested environment and discover local models without loading weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='')
    args=p.parse_args(argv)
    from model_registry import discover_models,resolve_model
    packages={};errors=[]
    for name in ['torch','torchvision','transformers','numpy','scipy','librosa','soundfile','opencv-python','music21','mido','matplotlib','requests','safetensors','sentencepiece']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None;errors.append(f'missing package: {name}')
    selected={}
    for family in ['mupt','musicgen']:
        try:selected[family]=resolve_model(family)
        except (ValueError,FileNotFoundError) as exc:errors.append(str(exc))
    report={'python':platform.python_version(),'packages':packages,'ffmpeg':shutil.which('ffmpeg'),
            'models':discover_models(),'selected_models':selected,'errors':errors,
            'verification':'package metadata and local checkpoint completeness, not generation or scientific validation'}
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    print(text);return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
