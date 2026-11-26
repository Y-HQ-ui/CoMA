"""Transparent training/repair proxies from produced artifacts, never paper scores."""
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from metrics.protocol import onset_f1


def score_artifacts(blueprint,symbolic_outputs,audio_manifest,*,consistency_enabled=True):
    from .evidence import extract_audio_features
    events=[e for item in symbolic_outputs for e in item.get("events",[])]
    if not events:raise ValueError("artifact scoring requires realized note events")
    pitches=np.asarray([e["pitch"] for e in events]);durations=np.asarray([e["duration"] for e in events])
    playable=float(np.mean((pitches>=21)&(pitches<=108)))
    diversity=min(1,len(np.unique(pitches%12))/7)
    duration=max(s.end_time for s in blueprint.segments)
    density=len({(e["track_id"],e["onset"]) for e in events})/duration
    requested=np.mean([t.density for s in blueprint.segments for t in s.tracks])*16
    density_fit=float(np.exp(-abs(density-requested)/max(1,requested)))
    symbolic=.4*playable+.3*diversity+.3*density_fit
    musicality=.5*playable+.25*diversity+.25*min(1,len(np.unique(np.round(durations,3)))/4)
    audio_path=audio_manifest.get("soundtrack_path") if audio_manifest else None
    consistency=visual=0.
    sources={"symbolic":"realized_score_events","audio":"not_generated"}
    if audio_path and Path(audio_path).exists():
        features=extract_audio_features(audio_path)
        onset=np.asarray(features["onset"]);hz=features["analysis_hz"]
        peaks,_=find_peaks(onset,height=.2,distance=max(1,int(.05*hz)))
        consistency=onset_f1(sorted({e["onset"] for e in events}),features["frame_times"][peaks],tolerance=.05)
        rms=np.asarray(features.get("rms_absolute",features["rms"]))
        target=np.zeros(len(rms));times=features["frame_times"]
        for s in blueprint.segments:target[(times>=s.start_time)&(times<s.end_time)]=s.energy
        # This is an explicitly labelled energy-envelope proxy, not EAS or RC.
        envelope=rms/(np.max(rms)+1e-8)
        visual=float(np.clip(1-np.mean(np.abs(envelope-target)),0,1)) if np.max(rms)>1e-5 else 0.
        sources["audio"]="waveform"
    components={"R_sym":float(symbolic),"R_mus":float(musicality),"R_vis":float(visual),"R_cons":float(consistency)}
    weights=np.asarray([.3,.3,.4 if consistency_enabled else 0]);weights/=weights.sum()
    components["R_total"]=float(weights @ np.asarray([visual,musicality,consistency]))
    return {"rewards":components,"evidence_source":sources,"scorer_version":"artifact_proxy_v1",
            "consistency_enabled":consistency_enabled,"consistency_definition":"one-to-one onset F1 at 50ms; no pitch/track fidelity claim"}


def weakest_audio_window(blueprint,symbolic_outputs,audio_manifest,*,window_seconds=.4):
    """Pick the lowest score/audio onset F1 window; never cross a segment boundary."""
    from .evidence import extract_audio_features
    if window_seconds<=0:raise ValueError("positive local repair window required")
    f=extract_audio_features(audio_manifest["soundtrack_path"])
    peaks,_=find_peaks(f["onset"],height=.2,distance=max(1,int(.05*f["analysis_hz"])))
    observed=f["frame_times"][peaks]
    expected=sorted({e["onset"] for item in symbolic_outputs for e in item.get("events",[])})
    windows=[]
    for index,segment in enumerate(blueprint.segments):
        for start in np.arange(segment.start_time,segment.end_time,window_seconds):
            end=min(segment.end_time,start+window_seconds)
            score=onset_f1([t for t in expected if start<=t<end],[t for t in observed if start<=t<end])
            windows.append({"segment_index":index,"start":float(start),"end":float(end),"onset_f1":score})
    return min(windows,key=lambda w:(w["onset_f1"],w["start"]))
