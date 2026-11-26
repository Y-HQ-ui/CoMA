"""Offline integration checks; outputs are smoke artifacts, never paper results.

Run as `python -m scripts.verify_model_adapters ...` from the project root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--audio-policy-dir',required=True)
    p.add_argument('--symbolic-checkpoint',required=True)
    p.add_argument('--candidate-dir',required=True)
    args=p.parse_args(argv);root=Path(args.output_dir)
    if root.exists() and any(root.iterdir()):raise FileExistsError('use a new output directory')
    root.mkdir(parents=True,exist_ok=True)
    from audio.musicgen_renderer import MusicGenRendererSession
    from audio.temporal_control import AdapterConfig,TemporalControlAdapter
    from model_registry import resolve_model
    from experiments.provenance import sha256_file
    audio_base=resolve_model('musicgen');symbolic_base=resolve_model('mupt')
    renderer=MusicGenRendererSession.get(model_dir=audio_base,device=args.device)
    zero=TemporalControlAdapter(AdapterConfig(renderer.model.decoder.config.hidden_size))
    zero_path=root/'zero_adapter.pt';zero.save(zero_path,metadata={'purpose':'zero-init identity check'})
    features=np.zeros((25,29),np.float32);features[:,0]=.4
    for label,checkpoint in [('baseline',''),('zero',str(zero_path)),('trained',str(Path(args.audio_policy_dir)/'temporal_adapter.pt'))]:
        renderer.render_one('instrumental piano',output_path=str(root/f'{label}.wav'),duration=.5,
            temporal_features=features,adapter_checkpoint=checkpoint,seed=907)
    a,sr=sf.read(root/'baseline.wav');b,_=sf.read(root/'zero.wav');c,_=sf.read(root/'trained.wav')
    if not np.array_equal(a,b):raise AssertionError('zero adapter changed pretrained model generation')
    if len(c)!=round(.5*sr) or not np.isfinite(c).all():raise AssertionError('trained adapter output invalid')
    trained=TemporalControlAdapter.load(Path(args.audio_policy_dir)/'temporal_adapter.pt')
    if trained.network[-1].weight.norm()==0:raise AssertionError('trained adapter is still zero')
    from planning import load_blueprint_json
    from experiments.realization import RealizationBackend
    from reward.refinement import Candidate,GenerationBudget
    candidate_dir=Path(args.candidate_dir)
    blueprint=load_blueprint_json(candidate_dir/'blueprint.json')
    symbolic=json.loads((candidate_dir/'symbolic_outputs.json').read_text())
    manifest=json.loads((candidate_dir/'audio/manifest.json').read_text())
    initial=Candidate(blueprint,symbolic,manifest,{},str(candidate_dir))
    backend=RealizationBackend(str(root/'local_repair'),musicgen_model_dir=audio_base,
        musicgen_device=args.device,adapter_checkpoint=str(Path(args.audio_policy_dir)/'temporal_adapter.pt'),local_audio_window_seconds=.4)
    budget=GenerationBudget(1);patched=backend.regenerate('audio',initial,907,budget)
    if not patched.audio_manifest['local_repair']['outside_mask_bit_identical']:raise AssertionError('local edit escaped its mask')
    if patched.symbolic_outputs!=symbolic or patched.blueprint.to_dict()!=blueprint.to_dict():raise AssertionError('local audio edit changed upstream artifacts')
    from track.melody.infer import generate
    prefix='X:1<n>M:4/4<n>L:1/8<n>Q:1/4=120<n>K:C<n>'
    kwargs=dict(model_dir=symbolic_base,device=args.device,policy_checkpoint=args.symbolic_checkpoint,
                max_new_tokens=48,seed=315,do_sample=True)
    first=generate(prefix,**kwargs);second=generate(prefix,**kwargs)
    if first!=second:raise AssertionError('cold/warm MuPT cache changed seeded generation')
    (root/'trained_symbolic.abc').write_text(first.replace('<n>','\n'))
    report={'status':'passed','purpose':'software smoke only; synthetic fixtures and one-step trained adapters',
        'musicgen_zero_adapter_bit_identical':True,'trained_audio_samples':len(c),'sample_rate':sr,
        'trained_adapter_output_weight_norm':float(trained.network[-1].weight.norm().detach()),
        'symbolic_checkpoint_loaded':args.symbolic_checkpoint,'symbolic_seeded_cache_identical':True,
        'local_audio_repair':patched.audio_manifest['local_repair'],'local_repair_generation_calls':budget.calls,
        'checkpoint_sha256':{'symbolic':sha256_file(args.symbolic_checkpoint),
                            'audio_adapter':sha256_file(Path(args.audio_policy_dir)/'temporal_adapter.pt')}}
    (root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
