#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五维先验特征包提取：
- 运动强度 motion_intensity (m_t)
- 剪辑密度 cut_density (c_t)
- 亮度 brightness (b_t)
- 色彩暖度 color_warmth (w_t)
- 视觉新颖度 visual_novelty (n_t)

说明：
- 该脚本不依赖网络，不调用LLM。
- 默认使用CV统计/相邻帧差分/颜色直方图。
- 显式提供 cineparse_checkpoint 时加载新重建的 ViT/GRU；实测 frame_features 保留用于评价。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Prior5DPoint:
    start_time: float
    end_time: float
    motion_intensity: float  # m_t
    cut_density: float  # c_t
    brightness: float  # b_t
    color_warmth: float  # w_t
    visual_novelty: float  # n_t


def _safe_norm01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _frame_brightness_bgr(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    return float(v.mean())


def _frame_color_warmth_bgr(frame_bgr: np.ndarray) -> float:
    """
    用HSV hue 粗略估计暖色程度：
    - 暖色（红/橙/黄）更高
    - 冷色（青/蓝）更低
    返回范围约 [0, 1]
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * 2.0  # OpenCV hue: [0,179] -> degrees [0,358]
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    # 只在有颜色信息的像素上统计（排除低饱和/低亮度）
    mask = (s > 0.2) & (v > 0.2)
    if not np.any(mask):
        return 0.5

    h = h[mask]
    # 暖色区间：[-30, 60] 近似等价于 [330,360] U [0,60]
    warm_dist = np.minimum(np.abs(h - 30.0), np.abs(h - 330.0))
    # 将距离映射为暖度（越接近暖色中心越暖）
    warmth = 1.0 - np.clip(warm_dist / 180.0, 0.0, 1.0)
    return float(np.mean(warmth))


def _hist_hsv(frame_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    hist = cv2.calcHist([h, s], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist.flatten()


def _motion_intensity(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    # 轻量：相邻帧差分能量
    diff = cv2.absdiff(prev_gray, gray).astype(np.float32) / 255.0
    return float(diff.mean())


def extract_prior_5d(
    video_path: str,
    *,
    segment_duration: float = 2.0,
    segment_overlap: float = 0.5,
    sample_fps: float = 2.0,
    cut_threshold: float = 0.35,
    cineparse_checkpoint: str = "",
    cineparse_backbone_checkpoint: str = "",
    cineparse_device: str = "cpu",
) -> Dict[str, object]:
    """
    提取五维先验特征包（按时间段输出）。
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if sample_fps <= 0 or not np.isfinite(sample_fps):
        raise ValueError("sample_fps must be finite and positive")
    if segment_duration <= 0 or not np.isfinite(segment_duration):
        raise ValueError("segment_duration must be positive")
    if not 0.0 <= segment_overlap < 1.0:
        raise ValueError("segment_overlap must be in [0, 1)")
    if not np.isfinite(cut_threshold) or not 0 <= cut_threshold <= 1:
        raise ValueError("cut_threshold must be finite and in [0,1]")
    if cineparse_backbone_checkpoint and not cineparse_checkpoint:
        raise ValueError("a trained cineparse_checkpoint is required with a backbone override")
    if cineparse_checkpoint and not Path(cineparse_checkpoint).is_file():
        raise FileNotFoundError(cineparse_checkpoint)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if fps > 0 else 0.0

    step = max(1, int(round((fps / sample_fps))) if fps > 0 else 1)

    times: List[float] = []
    motion: List[float] = []
    bright: List[float] = []
    warmth: List[float] = []
    novelty: List[float] = []
    cut_flags: List[int] = []

    prev_gray: Optional[np.ndarray] = None
    prev_hist: Optional[np.ndarray] = None

    frame_idx = 0
    sampled_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t = (frame_idx / fps) if fps > 0 else float(sampled_idx) / sample_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        b = _frame_brightness_bgr(frame)
        w = _frame_color_warmth_bgr(frame)
        h = _hist_hsv(frame, bins=24)

        if prev_gray is None:
            m = 0.0
        else:
            m = _motion_intensity(prev_gray, gray)

        if prev_hist is None:
            n = 0.0
            is_cut = 0
        else:
            # 视觉新颖度：颜色直方图差异
            # 使用 L1 距离并裁剪到 [0,1]
            n = float(np.clip(np.abs(h - prev_hist).sum() / 2.0, 0.0, 1.0))
            is_cut = 1 if n >= cut_threshold else 0

        times.append(t)
        motion.append(m)
        bright.append(b)
        warmth.append(w)
        novelty.append(n)
        cut_flags.append(is_cut)

        prev_gray = gray
        prev_hist = h

        frame_idx += 1
        sampled_idx += 1

    cap.release()

    if not times:
        raise ValueError("video contains no decodable sampled frames")

    if duration <= 0 and times:
        duration = float(times[-1])

    # 归一化到 [0,1]（便于直接进入提示词与后续模型）
    motion_n = _safe_norm01(np.array(motion))
    bright_n = np.clip(np.array(bright, dtype=np.float32), 0.0, 1.0)
    warmth_n = np.clip(np.array(warmth, dtype=np.float32), 0.0, 1.0)
    novelty_n = _safe_norm01(np.array(novelty))

    # 切分为 segment
    points: List[Prior5DPoint] = []
    if duration > 0:
        segment_stride = segment_duration * (1.0 - segment_overlap)
        num_seg = max(1, int(np.ceil(max(0.0, duration - segment_duration) / segment_stride)) + 1)
        for i in range(num_seg):
            st = i * segment_stride
            et = min(st + segment_duration, duration)

            idx = [k for k, tt in enumerate(times) if st <= tt < et]
            if not idx:
                continue

            mt = float(np.mean(motion_n[idx]))
            bt = float(np.mean(bright_n[idx]))
            wt = float(np.mean(warmth_n[idx]))
            nt = float(np.mean(novelty_n[idx]))

            # 剪辑密度：该段 cut 数 / 秒，压缩到 [0,1]
            cuts = int(np.sum(np.array(cut_flags, dtype=np.int32)[idx]))
            cd = cuts / max(1e-6, (et - st))
            ct = float(np.clip(cd / 2.0, 0.0, 1.0))  # 经验缩放：2 cuts/sec 视作 1.0

            points.append(
                Prior5DPoint(
                    start_time=st,
                    end_time=et,
                    motion_intensity=float(np.clip(mt, 0.0, 1.0)),
                    cut_density=float(np.clip(ct, 0.0, 1.0)),
                    brightness=float(np.clip(bt, 0.0, 1.0)),
                    color_warmth=float(np.clip(wt, 0.0, 1.0)),
                    visual_novelty=float(np.clip(nt, 0.0, 1.0)),
                )
            )

    summary = {
        "duration": float(duration),
        "segment_duration": float(segment_duration),
        "segment_overlap": float(segment_overlap),
        "segment_stride": float(segment_duration * (1.0 - segment_overlap)),
        "sample_fps": float(sample_fps),
        "cut_threshold": float(cut_threshold),
        "means": {
            "motion_intensity": float(np.mean(motion_n)) if len(motion_n) else 0.0,
            "cut_density": float(np.mean([p.cut_density for p in points])) if points else 0.0,
            "brightness": float(np.mean(bright_n)) if len(bright_n) else 0.0,
            "color_warmth": float(np.mean(warmth_n)) if len(warmth_n) else 0.0,
            "visual_novelty": float(np.mean(novelty_n)) if len(novelty_n) else 0.0,
        },
    }

    pack = {
        "video_path": str(path),
        "extractor": {"backend": "opencv_frame_statistics", "trainable": False,
                      "motion": "per_clip_minmax_frame_difference", "cuts": "HSV_histogram_L1_threshold",
                      "brightness": "mean_HSV_value", "warmth": "hue_distance_masked",
                      "novelty": "per_clip_minmax_HSV_histogram_L1"},
        "frame_features": {"times": times, "sample_hz": float(fps/step) if fps>0 else sample_fps,
                           "motion_intensity": motion_n.tolist(), "cut_flags": cut_flags,
                           "brightness": bright_n.tolist(), "color_warmth": warmth_n.tolist(),
                           "visual_novelty": novelty_n.tolist()},
        "summary": summary,
        "prior_5d": [asdict(p) for p in points],
    }
    if cineparse_checkpoint:
        from vedio.cineparse import apply_cineparse
        return apply_cineparse(pack, cineparse_checkpoint, backbone_path=cineparse_backbone_checkpoint,
                               device=cineparse_device)
    return pack


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="prior_feature_pack")
    parser.add_argument("video", help="视频路径")
    parser.add_argument("--segment", type=float, default=2.0, help="每段时长(秒)，论文设置为1-2秒")
    parser.add_argument("--segment-overlap", type=float, default=0.5, help="相邻分段重叠比例")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="采样帧率(Hz)")
    parser.add_argument("--cut-threshold", type=float, default=0.35, help="cut阈值(0-1)")
    parser.add_argument("--cineparse-checkpoint", default="", help="新重建 CineParse 的训练后检查点")
    parser.add_argument("--cineparse-backbone-checkpoint", default="", help="可选的本地 ViT 权重路径覆盖；必须匹配 SHA-256")
    parser.add_argument("--cineparse-device", default="cpu")
    parser.add_argument("--out", type=str, default="output/video_prior_5d.json", help="输出JSON路径")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pack = extract_prior_5d(
        args.video,
        segment_duration=args.segment,
        segment_overlap=args.segment_overlap,
        sample_fps=args.sample_fps,
        cut_threshold=args.cut_threshold,
        cineparse_checkpoint=args.cineparse_checkpoint,
        cineparse_backbone_checkpoint=args.cineparse_backbone_checkpoint,
        cineparse_device=args.cineparse_device,
    )

    out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

