from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class OnlinePolicy:
    """
    边推理边强化（在线 bandit）：
    - 将“推理时的采样超参组合”视为离散动作
    - 每生成一次就计算 reward，然后立刻更新策略偏好
    """

    action_specs: List[Dict[str, float]]
    logits: np.ndarray
    lr: float = 0.2

    @classmethod
    def create_default(cls, lr: float = 0.2) -> "OnlinePolicy":
        action_specs = [
            {"temperature": 0.6, "top_p": 0.90, "top_k": 250, "guidance_scale": 3.0},
            {"temperature": 0.8, "top_p": 0.95, "top_k": 250, "guidance_scale": 3.0},
            {"temperature": 1.0, "top_p": 0.98, "top_k": 250, "guidance_scale": 3.0},
            {"temperature": 0.9, "top_p": 0.92, "top_k": 150, "guidance_scale": 4.0},
        ]
        return cls(action_specs=action_specs, logits=np.zeros(len(action_specs), dtype=np.float32), lr=lr)

    def probs(self) -> np.ndarray:
        z = self.logits - float(np.max(self.logits))
        e = np.exp(z)
        return e / (float(e.sum()) + 1e-8)

    def sample_action(self, rng: np.random.Generator) -> int:
        p = self.probs()
        return int(rng.choice(len(p), p=p))

    def update(self, action: int, reward: float, baseline: float):
        p = self.probs()
        adv = float(reward - baseline)
        grad = -p
        grad[action] += 1.0
        self.logits = self.logits + self.lr * adv * grad.astype(np.float32)


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def load_video_prior_5d(prior_json_path: str) -> Dict[str, float]:
    """
    读取 `vedio/prior_feature_pack.py` 输出，并取 summary 均值作为全局目标。
    """
    data = json.loads(Path(prior_json_path).read_text(encoding="utf-8"))
    means = (data.get("summary") or {}).get("means") or {}
    return {
        "motion_intensity": float(means.get("motion_intensity", 0.5)),
        "cut_density": float(means.get("cut_density", 0.5)),
        "brightness": float(means.get("brightness", 0.5)),
        "color_warmth": float(means.get("color_warmth", 0.5)),
        "visual_novelty": float(means.get("visual_novelty", 0.5)),
    }


def load_prompts_from_file(prompt_path: str) -> List[str]:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {prompt_path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def load_prompts_from_render_controls(render_controls_json: str) -> List[str]:
    from audio.musicgen_adapter import render_controls_to_prompts

    path = Path(render_controls_json)
    if not path.exists():
        raise FileNotFoundError(f"render controls file not found: {render_controls_json}")
    controls = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(controls, list):
        raise ValueError("render controls json must contain a list")
    return render_controls_to_prompts(controls)


def compute_audio_proxy_features(wav: np.ndarray, sr: int) -> Dict[str, float]:
    """
    用音频代理特征对齐视频 5D 先验（均归一化到 0-1）。
    """
    import librosa  # 延迟导入

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1 or wav.size < sr // 4:
        return {"rhythm": 0.0, "brightness": 0.0, "novelty": 0.0, "energy": 0.0}

    hop = 512
    rms = librosa.feature.rms(y=wav, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=wav, sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr, hop_length=hop)[0]

    energy = float(np.mean(rms))
    rhythm = float(np.mean(onset))
    bright = float(np.mean(centroid))
    novelty = float(np.std(centroid) + np.std(onset))

    return {
        "energy": _norm01(energy, 0.01, 0.15),
        "rhythm": _norm01(rhythm, 0.1, 5.0),
        "brightness": _norm01(bright, 800.0, 4000.0),
        "novelty": _norm01(novelty, 50.0, 800.0),
    }


def compute_reward_from_video_prior(*, audio_feat: Dict[str, float], video_prior: Dict[str, float]) -> float:
    """
    在线 reward：让生成音频的代理特征贴近视频先验（简单可用版本）。
    """
    target_energy = video_prior["motion_intensity"]
    target_rhythm = 0.6 * video_prior["motion_intensity"] + 0.4 * video_prior["cut_density"]
    target_brightness = 0.6 * video_prior["brightness"] + 0.4 * video_prior["color_warmth"]
    target_novelty = video_prior["visual_novelty"]

    def close(x, t):
        return 1.0 - abs(float(x) - float(t))

    r = 0.0
    r += 0.30 * close(audio_feat["energy"], target_energy)
    r += 0.30 * close(audio_feat["rhythm"], target_rhythm)
    r += 0.25 * close(audio_feat["brightness"], target_brightness)
    r += 0.15 * close(audio_feat["novelty"], target_novelty)
    return float(np.clip(r, 0.0, 1.0))


def save_spectrogram(wav: np.ndarray, sr: int, out_png: Path):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    S = librosa.stft(wav, n_fft=1024, hop_length=256, win_length=1024)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def generate_with_online_rl(
    *,
    prompts: List[str],
    num_variants_per_prompt: int,
    model_dir: str,
    out_audio_dir: str = "audio",
    out_spec_dir: str = "spectrograms",
    seed: int = 123,
    video_prior_json: Optional[str] = None,
    policy_lr: float = 0.2,
    save_policy_log: str = "output/online_rl_log.jsonl",
) -> None:
    """
    边推理边强化：每次生成一个 variant 就立刻计算 reward 并更新 policy。
    """
    import torch
    import soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    model = MusicgenForConditionalGeneration.from_pretrained(model_dir).to(device)
    processor = AutoProcessor.from_pretrained(model_dir)

    os.makedirs(out_audio_dir, exist_ok=True)
    os.makedirs(out_spec_dir, exist_ok=True)
    Path(save_policy_log).parent.mkdir(parents=True, exist_ok=True)

    video_prior = load_video_prior_5d(video_prior_json) if video_prior_json else {
        "motion_intensity": 0.5,
        "cut_density": 0.5,
        "brightness": 0.5,
        "color_warmth": 0.5,
        "visual_novelty": 0.5,
    }

    policy = OnlinePolicy.create_default(lr=policy_lr)
    baseline = 0.0
    beta = 0.05  # baseline EMA

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sr = int(model.config.audio_encoder.sampling_rate)

    with open(save_policy_log, "w", encoding="utf-8") as f:
        for pid, prompt in enumerate(prompts):
            for vid in range(num_variants_per_prompt):
                action = policy.sample_action(rng)
                spec = policy.action_specs[action]

                print(f"Generating: prompt {pid}, variant {vid}, action {action}, spec={spec}")

                inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=1500,
                        do_sample=True,
                        temperature=float(spec["temperature"]),
                        top_k=int(spec["top_k"]),
                        top_p=float(spec["top_p"]),
                        guidance_scale=float(spec["guidance_scale"]),
                    )

                audio = audio_values[0].detach().cpu().numpy()
                wav = audio[0] if (audio.ndim == 2 and audio.shape[0] == 1) else audio.mean(axis=0)

                audio_feat = compute_audio_proxy_features(wav, sr)
                reward = compute_reward_from_video_prior(audio_feat=audio_feat, video_prior=video_prior)

                baseline = (1.0 - beta) * baseline + beta * reward
                policy.update(action=action, reward=reward, baseline=baseline)

                wav_path = Path(out_audio_dir) / f"prompt{pid}_var{vid}.wav"
                sf.write(str(wav_path), wav, sr)

                spec_path = Path(out_spec_dir) / f"prompt{pid}_var{vid}.png"
                save_spectrogram(wav, sr, spec_path)

                f.write(
                    json.dumps(
                        {
                            "pid": pid,
                            "vid": vid,
                            "prompt": prompt,
                            "action": action,
                            "action_spec": spec,
                            "audio_feat": audio_feat,
                            "video_prior": video_prior,
                            "reward": reward,
                            "baseline": baseline,
                            "policy": {"logits": policy.logits.tolist(), "probs": policy.probs().tolist()},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                del inputs, audio_values
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("全部生成完成（边推理边强化）！")


def generate_from_prompt_list(
    *,
    prompts: List[str],
    model_dir: str,
    out_audio_dir: str = "audio",
    out_spec_dir: str = "spectrograms",
    seed: int = 123,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 250,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 1500,
) -> None:
    import torch
    import soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicgenForConditionalGeneration.from_pretrained(model_dir).to(device)
    processor = AutoProcessor.from_pretrained(model_dir)

    os.makedirs(out_audio_dir, exist_ok=True)
    os.makedirs(out_spec_dir, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sr = int(model.config.audio_encoder.sampling_rate)
    for idx, prompt in enumerate(prompts):
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                guidance_scale=float(guidance_scale),
            )
        audio = audio_values[0].detach().cpu().numpy()
        wav = audio[0] if (audio.ndim == 2 and audio.shape[0] == 1) else audio.mean(axis=0)
        wav_path = Path(out_audio_dir) / f"segment_{idx}.wav"
        sf.write(str(wav_path), wav, sr)
        spec_path = Path(out_spec_dir) / f"segment_{idx}.png"
        save_spectrogram(wav, sr, spec_path)
        del inputs, audio_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="infer")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.getenv("MUSICAGENT_MUSICGEN_MODEL_DIR", ""),
        help="MusicGen 模型目录；使用音频生成时必须提供",
    )
    parser.add_argument("--prompt", type=str, action="append", default=["Epic orchestral adventure, heroic and powerful"])
    parser.add_argument("--num-variants", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--video-prior-json", type=str, default=None, help="`vedio/prior_feature_pack.py` 输出的 JSON")
    parser.add_argument("--prompt-file", type=str, default=None, help="每行一个 MusicGen prompt")
    parser.add_argument("--render-controls-json", type=str, default=None, help="run_coma_pipeline 输出的 render_controls JSON")
    parser.add_argument("--use-refined-prompts", action="store_true", help="优先使用 refined prompts 文件")
    parser.add_argument("--refined-prompts-file", type=str, default=None, help="run_coma_pipeline 输出的 refined prompts txt")
    parser.add_argument("--policy-lr", type=float, default=0.2)
    parser.add_argument("--out-audio-dir", type=str, default="audio")
    parser.add_argument("--out-spec-dir", type=str, default="spectrograms")
    parser.add_argument("--policy-log", type=str, default="output/online_rl_log.jsonl")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    args = parser.parse_args(argv)

    prompts = args.prompt or []
    if args.use_refined_prompts and args.refined_prompts_file:
        prompts = load_prompts_from_file(args.refined_prompts_file)
    elif args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
    elif args.render_controls_json:
        prompts = load_prompts_from_render_controls(args.render_controls_json)

    if prompts and (args.prompt_file or args.render_controls_json or (args.use_refined_prompts and args.refined_prompts_file)):
        generate_from_prompt_list(
            prompts=prompts,
            model_dir=args.model_dir,
            out_audio_dir=args.out_audio_dir,
            out_spec_dir=args.out_spec_dir,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            guidance_scale=args.guidance_scale,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        generate_with_online_rl(
            prompts=args.prompt,
            num_variants_per_prompt=args.num_variants,
            model_dir=args.model_dir,
            out_audio_dir=args.out_audio_dir,
            out_spec_dir=args.out_spec_dir,
            seed=args.seed,
            video_prior_json=args.video_prior_json,
            policy_lr=args.policy_lr,
            save_policy_log=args.policy_log,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

