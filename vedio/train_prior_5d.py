from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Sample:
    video: str
    y: np.ndarray


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_samples(data_path: str) -> List[Sample]:
    path = Path(data_path)
    rows = list(_read_jsonl(path))
    samples = []
    names = ("motion_intensity", "cut_density", "brightness", "color_warmth", "visual_novelty")
    for i, row in enumerate(rows):
        if row.get("split") != "train":
            raise ValueError(f"row {i}: prior calibration requires split=train")
        video = Path(row["video"])
        if not video.is_absolute(): video = path.parent / video
        if not video.is_file(): raise FileNotFoundError(video)
        target = row.get("y", row.get("prior_5d"))
        if isinstance(target, dict): target = [target[k] for k in names]
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (5,) or not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
            raise ValueError(f"row {i}: labels must be five finite values in [0,1]")
        samples.append(Sample(str(video.resolve()), target))
    if not samples: raise ValueError("empty prior calibration dataset")
    return samples


def _warmth(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32) * 2.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    mask = (s > 0.2) & (v > 0.2)
    if not np.any(mask):
        return 0.5
    h = h[mask]
    warm_dist = np.minimum(np.abs(h - 30.0), np.abs(h - 330.0))
    warmth = 1.0 - np.clip(warm_dist / 180.0, 0.0, 1.0)
    return float(np.mean(warmth))


def _hist_hs(frame_bgr: np.ndarray, bins: int = 24) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    hist = cv2.calcHist([h, s], [0, 1], None, [bins, bins], [0, 180, 0, 256]).astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist.flatten()


def _extract_x(
    video_path: str,
    *,
    sample_fps: float,
    max_frames: int,
    resize: int,
    cut_threshold: float,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(1, int(round((fps / sample_fps))) if fps > 0 else 1)

    prev_gray: Optional[np.ndarray] = None
    prev_hist: Optional[np.ndarray] = None

    motion_vals: List[float] = []
    bright_vals: List[float] = []
    warm_vals: List[float] = []
    nov_vals: List[float] = []
    cut_vals: List[float] = []

    frame_idx = 0
    kept = 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        if resize > 0:
            h, w = frame.shape[:2]
            if max(h, w) > resize:
                scale = resize / float(max(h, w))
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].astype(np.float32) / 255.0

        bright_vals.append(float(v.mean()))
        warm_vals.append(_warmth(frame))

        hst = _hist_hs(frame)
        if prev_hist is None:
            n = 0.0
            cut = 0.0
        else:
            n = float(np.clip(np.abs(hst - prev_hist).sum() / 2.0, 0.0, 1.0))
            cut = 1.0 if n >= cut_threshold else 0.0
        nov_vals.append(n)
        cut_vals.append(cut)
        prev_hist = hst

        if prev_gray is None:
            motion_vals.append(0.0)
        else:
            diff = cv2.absdiff(prev_gray, gray).astype(np.float32) / 255.0
            motion_vals.append(float(diff.mean()))
        prev_gray = gray

        kept += 1
        frame_idx += 1

    cap.release()

    def stats(a: List[float]) -> Tuple[float, float]:
        if not a:
            return 0.0, 0.0
        x = np.asarray(a, dtype=np.float32)
        return float(x.mean()), float(x.std())

    m_mu, m_sd = stats(motion_vals)
    b_mu, b_sd = stats(bright_vals)
    w_mu, w_sd = stats(warm_vals)
    n_mu, n_sd = stats(nov_vals)
    c_mu = float(np.mean(cut_vals)) if cut_vals else 0.0
    c_sum = float(np.sum(cut_vals)) if cut_vals else 0.0

    x = np.array(
        [m_mu, m_sd, c_mu, c_sum, b_mu, b_sd, w_mu, w_sd, n_mu, n_sd, 1.0],
        dtype=np.float32,
    )
    return x


def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float32)
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    return W.astype(np.float32)


def train(
    *,
    data_path: str,
    sample_fps: float,
    max_frames: int,
    resize: int,
    cut_threshold: float,
    lam: float,
) -> Dict[str, object]:
    samples = _load_samples(data_path)

    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    for s in samples:
        Xs.append(_extract_x(s.video, sample_fps=sample_fps, max_frames=max_frames, resize=resize, cut_threshold=cut_threshold))
        Ys.append(s.y)

    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.float32)

    x_mu = X.mean(axis=0, keepdims=True)
    x_sd = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - x_mu) / x_sd
    Xn[:, -1] = 1.0  # preserve intercept after feature standardization

    W = _ridge_fit(Xn, Y, lam=lam)
    pred = Xn @ W
    mse = float(np.mean((pred - Y) ** 2))

    return {
        "ok": True,
        "backend": "supervised_ridge_on_opencv_statistics",
        "version": 2,
        "supervision": "five annotated values in [0,1] per video",
        "metric_scope": "training_fit_only_not_validation",
        "n": int(X.shape[0]),
        "x_dim": int(X.shape[1]),
        "y_dim": int(Y.shape[1]),
        "mse": mse,
        "norm": {"mu": x_mu.reshape(-1).tolist(), "sd": x_sd.reshape(-1).tolist()},
        "W": W.tolist(),
        "config": {
            "sample_fps": float(sample_fps),
            "max_frames": int(max_frames),
            "resize": int(resize),
            "cut_threshold": float(cut_threshold),
            "lam": float(lam),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="train_prior_5d")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=os.getenv("MUSICAGENT_PRIOR_MODEL_OUT", "output/prior5d_ridge.json"))
    parser.add_argument("--sample-fps", type=float, default=float(os.getenv("MUSICAGENT_PRIOR_SAMPLE_FPS", "2.0")))
    parser.add_argument("--max-frames", type=int, default=int(os.getenv("MUSICAGENT_PRIOR_MAX_FRAMES", "240")))
    parser.add_argument("--resize", type=int, default=int(os.getenv("MUSICAGENT_PRIOR_RESIZE", "384")))
    parser.add_argument("--cut-threshold", type=float, default=float(os.getenv("MUSICAGENT_PRIOR_CUT_TH", "0.35")))
    parser.add_argument("--lam", type=float, default=float(os.getenv("MUSICAGENT_PRIOR_LAM", "1.0")))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    result = train(
        data_path=args.data,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        resize=args.resize,
        cut_threshold=args.cut_threshold,
        lam=args.lam,
    )
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

