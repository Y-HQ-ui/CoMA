"""Paper-formula implementations with explicitly chosen reconstruction conventions."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import rankdata


def correlation(x, y, *, rank=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("correlation needs aligned finite vectors of length >=3")
    if rank:
        x, y = rankdata(x), rankdata(y)
    x, y = x-x.mean(), y-y.mean()
    scale = np.linalg.norm(x)*np.linalg.norm(y)
    if scale <= 1e-12:
        raise ValueError("correlation is undefined for constant curves")
    return float(np.clip(x@y/scale, -1, 1))


def emotion_alignment(video_va, audio_va, *, sample_hz, max_lag_seconds):
    """Max over one shared lag of mean Spearman(V,A), reranking each overlap."""
    x, y = np.asarray(video_va, float), np.asarray(audio_va, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2 or len(x) < 3 or not np.isfinite([x,y]).all():
        raise ValueError("EAS requires aligned finite [T>=3,2] VA trajectories")
    if sample_hz <= 0 or max_lag_seconds < 0:
        raise ValueError("invalid EAS sampling/lag")
    bound = min(math.floor(sample_hz*max_lag_seconds), len(x)-3)
    scores = []
    for lag in range(-bound, bound+1):
        a,b = (x[:-lag],y[lag:]) if lag>0 else (x[-lag:],y[:lag]) if lag<0 else (x,y)
        try:
            scores.append((float(np.mean([correlation(a[:,i],b[:,i],rank=True) for i in range(2)])), lag/sample_hz))
        except ValueError:
            pass
    if not scores:
        raise ValueError("EAS has no nonconstant VA overlap")
    return max(scores, key=lambda item:(item[0],-abs(item[1])))


def color_timbre_alignment(brightness, warmth, centroid, slope):
    return .5*(correlation(brightness,centroid)+correlation(warmth,-np.asarray(slope)))


def chroma_diversity(matrices):
    """1 - average pairwise cosine of equal-shaped flattened segment chroma."""
    x = np.asarray(matrices, float)
    if x.ndim != 3 or x.shape[1] != 12 or x.shape[0] < 2 or x.shape[2] < 1 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError("Chroma Diversity requires [N>=2,12,T] nonnegative matrices")
    flat = x.reshape(len(x),-1).copy()
    norms = np.linalg.norm(flat, axis=1)
    if (norms<=1e-12).any():
        raise ValueError("silent chroma segment has undefined similarity")
    flat /= norms[:,None]
    return float(1-np.clip((flat@flat.T)[np.triu_indices(len(x),1)],0,1).mean())


def harmonic_consistency(tonal_centroids, *, sigma_tc):
    tc = np.asarray(tonal_centroids,float)
    if tc.ndim != 2 or tc.shape[1] != 6 or len(tc)<2 or not np.isfinite(tc).all() or not np.isfinite(sigma_tc) or sigma_tc<=0:
        raise ValueError("HCS requires finite [T>=2,6] tonal centroids and explicit positive sigma_TC")
    # The paper formula is not clipped to [0,1]. A negative value is informative.
    return float(1-np.linalg.norm(np.diff(tc,axis=0),axis=1).mean()/sigma_tc)


def dynamic_variation(reference_db, generated_db):
    ref, gen = np.asarray(reference_db,float),np.asarray(generated_db,float)
    if ref.ndim != 1 or ref.shape != gen.shape or len(ref)<2 or not np.isfinite([ref,gen]).all():
        raise ValueError("DV requires aligned finite decibel trajectories")
    scale = np.linalg.norm(ref)*np.linalg.norm(gen)
    if scale <= 1e-12:
        raise ValueError("zero-norm dB trajectory has undefined cosine")
    return float(1-np.clip(ref@gen/scale,-1,1))


def waveform_features(waveform, sr, *, n_fft=2048, hop_length=512, silence_floor_db=-80.):
    import librosa
    y=np.asarray(waveform,float)
    if y.ndim != 1 or len(y)<n_fft or not np.isfinite(y).all():
        raise ValueError("finite mono waveform of at least one analysis window required")
    spectrum=np.abs(librosa.stft(y,n_fft=n_fft,hop_length=hop_length,center=False))
    freqs=librosa.fft_frequencies(sr=sr,n_fft=n_fft)
    mass=spectrum.sum(axis=0)
    norm=spectrum/np.maximum(mass,1e-12)
    centered=freqs-freqs.mean()
    slope=centered@norm/np.sum(centered**2)
    rms=librosa.feature.rms(y=y,frame_length=n_fft,hop_length=hop_length,center=False)[0]
    chroma=librosa.feature.chroma_stft(S=spectrum**2,sr=sr,n_fft=n_fft,hop_length=hop_length)
    return {"times":(np.arange(len(rms))*hop_length+n_fft/2)/sr,
            "centroid":freqs@norm,"slope":slope,"chroma":chroma,
            "rms_db":20*np.log10(np.maximum(rms,10**(silence_floor_db/20))),
            "active":rms>10**(silence_floor_db/20),
            "protocol":{"n_fft":n_fft,"hop_length":hop_length,"center":False,"sample_rate":sr,
                        "slope":"OLS of L1-normalized magnitude on frequency Hz", "db_reference":"full_scale_amplitude_1",
                        "silence_floor_db":silence_floor_db,"channel_policy":"mono_average"}}


def segment_chroma(features, *, duration, segment_seconds=2., frames_per_segment=32):
    if segment_seconds <= 0 or frames_per_segment < 1:
        raise ValueError("invalid chroma segment protocol")
    segments=[]
    n=math.floor(duration/segment_seconds+1e-8)
    for i in range(n):
        start,end=i*segment_seconds,(i+1)*segment_seconds
        mask=(features["times"]>=start)&(features["times"]<end)
        if not mask.any() or not features["active"][mask].any():
            raise ValueError("missing or silent chroma segment")
        source=features["chroma"][:,mask]
        grid=np.linspace(0,1,frames_per_segment)
        segments.append(np.stack([np.interp(grid,np.linspace(0,1,source.shape[1]),pc) for pc in source]))
    if len(segments)<2:
        raise ValueError("Chroma Diversity needs at least two complete segments")
    return segments
