import argparse
import json
import time
import platform
from pathlib import Path
from planning import load_blueprint_json
from planning.timeline import merge_overlapping_segments
from reward.refinement import GenerationBudget, refine_inference
from runtime_protocol import REWARD_CONFIG, build_refinement, proposal_settings
from .realization import RealizationBackend, save_initial, load_initial
from .provenance import sha256_file, source_fingerprint
STRATEGIES = ['none', 'best_of_n', 'audio_only', 'random', 'full', 'routed']

def compare_strategies(initial, initial_snapshot, backend_factory, *, strategies, seed, max_calls=None, max_seconds=None, iterations=5, stopping='fixed', routing_seed=2026):
    records = []
    for strategy in strategies:
        budget = GenerationBudget.from_snapshot(initial_snapshot['budget'], max_calls, max_seconds)
        backend = backend_factory(strategy)
        try:
            final, trace = refine_inference(initial, backend.regenerate, budget=budget, strategy=strategy, max_iterations=iterations, stopping=stopping, patience=5 if stopping == 'convergence' else 2, seed=seed, routing_seed=routing_seed)
            record = {'seed': seed, 'strategy': strategy, 'status': 'ok', 'initial_calls': initial_snapshot['budget']['calls'], 'initial_snapshot_sha256': initial_snapshot['snapshot_sha256'], 'initial_artifact_dir': initial.artifact_dir, 'initial_rewards': initial.rewards, 'initial_cost_origin': initial_snapshot['origin'], 'final_rewards': final.rewards, 'final_audio': final.audio_manifest, 'final_artifact_dir': final.artifact_dir, 'trace': trace}
        except Exception as exc:
            record = {'seed': seed, 'strategy': strategy, 'status': 'failed', 'error': str(exc), 'calls': budget.calls, 'elapsed_seconds': time.monotonic() - budget.started}
        records.append(record)
    return records

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--blueprint', help='saved blueprint; supply --composer-trace to account for its creation')
    inputs.add_argument('--initial-candidate', help='reuse a previously saved initial_candidate.json (one seed)')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--strategies', nargs='+', default=STRATEGIES, choices=STRATEGIES)
    p.add_argument('--seeds', nargs='+', type=int, default=[123])
    p.add_argument('--routing-seed', type=int, default=2026)
    p.add_argument('--max-generation-calls', type=int)
    p.add_argument('--max-seconds', type=float)
    p.add_argument('--max-iterations', type=int, default=5)
    p.add_argument('--proposal-mode', choices=['adaptive', 'refine-5', 'refine-30', 'convergence'], default='refine-5')
    p.add_argument('--mupt-model-dir', default='')
    p.add_argument('--musicgen-model-dir', default='')
    p.add_argument('--mupt-device', default='cpu')
    p.add_argument('--musicgen-device', default='cpu')
    p.add_argument('--adapter-checkpoint', default='')
    p.add_argument('--symbolic-policy-checkpoint', default='')
    p.add_argument('--symbolic-max-new-tokens', type=int, default=512)
    p.add_argument('--local-audio-window-seconds', type=float, default=0.4)
    p.add_argument('--reward-inputs', required=True)
    p.add_argument('--reward-config', default=REWARD_CONFIG)
    p.add_argument('--composer-prior-json', default='', help='observed CineParse priors; also generate the initial blueprint if none supplied')
    p.add_argument('--composer-trace', default='', help='saved initial Composer trace with generation_budget snapshot')
    p.add_argument('--user-prompt', default='')
    p.add_argument('--reference-audio', default='')
    args = p.parse_args(argv)
    if len(set(args.strategies)) != len(args.strategies) or len(set(args.seeds)) != len(args.seeds):
        p.error('strategies and seeds must be unique')
    if not (args.blueprint or args.initial_candidate or args.composer_prior_json):
        p.error('supply observed Composer priors, a saved blueprint, or an initial candidate')
    if args.initial_candidate and len(args.seeds) != 1:
        p.error('a saved initial candidate belongs to one seed')
    if args.composer_trace and (not args.blueprint):
        p.error('--composer-trace accompanies --blueprint')
    if set(args.strategies) & {'routed', 'random', 'full'} and (not args.composer_prior_json):
        p.error('blueprint proposals require --composer-prior-json')
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('use a new output directory')
    iterations, stopping = proposal_settings(args.proposal_mode, args.max_iterations)
    prior = json.loads(Path(args.composer_prior_json).read_text()) if args.composer_prior_json else None
    if prior and prior.get('video_path') and (not Path(prior['video_path']).is_absolute()):
        prior['video_path'] = str((Path(args.composer_prior_json).resolve().parent / prior['video_path']).resolve())
    reference = None
    if args.reference_audio:
        from audio.reference_audio import extract_reference_audio_features
        reference = extract_reference_audio_features(args.reference_audio)
    scorer, planner = build_refinement(args.reward_inputs, prior, args.reward_config, user_prompt=args.user_prompt, reference_audio=reference)
    from model_registry import resolve_model
    mupt = resolve_model('mupt', args.mupt_model_dir)
    musicgen = resolve_model('musicgen', args.musicgen_model_dir)
    from track.melody.infer import load_model
    from audio.musicgen_renderer import MusicGenRendererSession
    load_started = time.monotonic()
    load_model(mupt, args.mupt_device, policy_checkpoint=args.symbolic_policy_checkpoint)
    renderer = MusicGenRendererSession.get(model_dir=musicgen, device=args.musicgen_device)
    if args.adapter_checkpoint:
        renderer.get_adapter(args.adapter_checkpoint)
    scorer.preload()
    import torch
    hardware = {'platform': platform.platform(), 'torch': str(torch.__version__), 'batch_size': 1, 'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'model_load_seconds': time.monotonic() - load_started}
    root.mkdir(parents=True, exist_ok=True)
    binding = {'reward_inputs_sha256': sha256_file(args.reward_inputs), 'reward_config_sha256': sha256_file(args.reward_config), 'base_models': {'mupt': mupt, 'musicgen': musicgen}, 'symbolic_checkpoint_sha256': sha256_file(args.symbolic_policy_checkpoint) if args.symbolic_policy_checkpoint else None, 'adapter_checkpoint_sha256': sha256_file(args.adapter_checkpoint) if args.adapter_checkpoint else None}

    def backend(directory, strategy='routed'):
        return RealizationBackend(str(directory), mupt, musicgen, args.mupt_device, args.musicgen_device, args.adapter_checkpoint, independent_sampling=strategy == 'best_of_n', symbolic_policy_checkpoint=args.symbolic_policy_checkpoint, max_new_tokens=args.symbolic_max_new_tokens, local_audio_window_seconds=args.local_audio_window_seconds, scorer=scorer, planner=planner)
    records = []
    for seed in args.seeds:
        seed_root = root / f'seed_{seed}'
        budget = None
        try:
            if args.initial_candidate:
                initial, snapshot = load_initial(args.initial_candidate)
                if snapshot['seed'] != seed:
                    raise ValueError('initial candidate seed differs')
                if snapshot['origin'].get('binding') != binding:
                    raise ValueError('initial candidate scoring/model configuration differs')
            else:
                budget = GenerationBudget(args.max_generation_calls, args.max_seconds)
                origin = {'scope': 'all_initial_generation_calls', 'binding': binding}
                if args.blueprint:
                    blueprint = merge_overlapping_segments(load_blueprint_json(args.blueprint))
                    if args.composer_trace:
                        trace = json.loads(Path(args.composer_trace).read_text())
                        from .provenance import canonical_hash
                        if trace.get('blueprint_sha256') != canonical_hash(blueprint.to_dict()):
                            raise ValueError('Composer trace is not bound to the supplied blueprint')
                        budget = GenerationBudget.from_snapshot(trace['generation_budget'], args.max_generation_calls, args.max_seconds)
                        origin['composer_trace_sha256'] = sha256_file(args.composer_trace)
                    else:
                        origin['scope'] = 'from_saved_blueprint; original Composer cost unavailable'
                else:
                    from main import BlueprintComposerController
                    import copy
                    composer = BlueprintComposerController()
                    blueprint = merge_overlapping_segments(composer.plan_blueprint(copy.deepcopy(prior), budget=budget, user_prompt=args.user_prompt, reference_audio=reference))
                    seed_root.mkdir(parents=True, exist_ok=True)
                    (seed_root / 'composer_trace.json').write_text(json.dumps(composer.last_trace, indent=2))
                initial = backend(seed_root / 'initial').generate(blueprint, seed=seed, budget=budget)
                snapshot = save_initial(initial, budget, seed_root / 'initial_candidate.json', seed=seed, origin=origin)
            records.extend(compare_strategies(initial, snapshot, lambda strategy: backend(seed_root / strategy, strategy), strategies=args.strategies, seed=seed, max_calls=args.max_generation_calls, max_seconds=args.max_seconds, iterations=iterations, stopping=stopping, routing_seed=args.routing_seed))
        except Exception as exc:
            records.extend(({'seed': seed, 'strategy': strategy, 'status': 'failed', 'phase': 'initialization', 'error': str(exc), 'initial_budget': budget.snapshot() if budget is not None else None} for strategy in args.strategies))
        (root / 'results.json').write_text(json.dumps({'protocol': vars(args), 'records': records, 'hardware': hardware, 'scope': 'frozen six-component acceptance; shared initial realization per generation seed', 'timing': 'warm models; each strategy charged original initial cost plus its own generation/retries/IO/scoring; atomic calls may exceed deadline', 'source': source_fingerprint(Path(__file__).resolve().parents[1])}, indent=2, allow_nan=False))
    return 0 if all((r['status'] == 'ok' for r in records)) else 1
if __name__ == '__main__':
    raise SystemExit(main())
