"""Explicit v2 evaluation protocol. Missing scientific evaluators stay unavailable.

No metric is computed from a blueprint as a substitute for generated music.
Feature-based diagnostics are labelled separately from paper metric names.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

VERSION = "coma_eval_v2"


def lagged_ncc(left, right, *, sample_hz, max_lag_seconds=.1):
    """Max zero-mean NCC over |lag|<=floor(hz*lag_s), no circular wrapping.

    Constant curves have undefined NCC and raise ValueError. Positive lag means
    the right/audio sequence is delayed. Values are in [-1,1], not probabilities.
    """
    x,y=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("NCC requires aligned finite 1D curves of length >=3")
    if sample_hz <= 0 or max_lag_seconds < 0: raise ValueError("invalid NCC resolution/lag")
    bound=min(math.floor(sample_hz*max_lag_seconds),len(x)-3)
    scores=[]
    for lag in range(-bound,bound+1):
        a,b=(x[:-lag],y[lag:]) if lag>0 else (x[-lag:],y[:lag]) if lag<0 else (x,y)
        a=a-a.mean(); b=b-b.mean(); denominator=np.linalg.norm(a)*np.linalg.norm(b)
        if denominator>1e-12: scores.append((float(np.clip(a@b/denominator,-1,1)),lag/sample_hz))
    if not scores: raise ValueError("NCC is undefined for constant signals")
    return max(scores,key=lambda v:(v[0],-abs(v[1])))


def imagebind_score(video, audio, *, candidates=None, target_index=None, temperature=None):
    video=np.asarray(video,dtype=float).reshape(-1); audio=np.asarray(audio,dtype=float).reshape(-1)
    if video.shape != audio.shape or not np.isfinite(video).all() or not np.isfinite(audio).all():
        raise ValueError("ImageBind dimensions must match exactly and be finite")
    def unit(x):
        norm=np.linalg.norm(x,axis=-1,keepdims=True)
        if (norm<=1e-12).any(): raise ValueError("zero embedding has undefined cosine")
        return x/norm
    if candidates is None: return float(unit(video)@unit(audio))
    candidates=np.asarray(candidates,dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != len(video) or not np.isfinite(candidates).all():
        raise ValueError("invalid candidate embeddings")
    if temperature is None or temperature<=0 or target_index is None or not 0<=target_index<len(candidates):
        raise ValueError("retrieval probability needs positive temperature and matched candidate index")
    if not np.allclose(candidates[target_index],audio): raise ValueError("matched candidate differs from audio embedding")
    logits=unit(candidates)@unit(video)/temperature
    probs=np.exp(logits-logits.max()); probs/=probs.sum()
    return float(probs[target_index])


def frechet_embeddings(generated, reference):
    from .paper_metrics import _frechet_distance
    x,y=np.asarray(generated,dtype=float),np.asarray(reference,dtype=float)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y))<2:
        raise ValueError("FAD requires >=2 generated/reference embeddings with equal dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError("nonfinite embedding")
    return max(0.0,_frechet_distance(x,y))


def inception_score(probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or min(p.shape)<2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
        raise ValueError("IS needs finite, normalized categorical class probabilities [N>=2,C>=2]; multilabel scores are not normalized implicitly")
    p=np.clip(p,1e-12,1); marginal=p.mean(axis=0)
    return float(np.exp((p*(np.log(p)-np.log(marginal))).sum(axis=1).mean()))


def onset_f1(symbolic_times, audio_times, tolerance=.05):
    """One-to-one event matching. Silence alone is not positive evidence."""
    left=sorted(map(float,symbolic_times)); right=sorted(map(float,audio_times))
    if not left and not right: return 0.0
    i=j=matched=0
    while i<len(left) and j<len(right):
        d=left[i]-right[j]
        if abs(d)<=tolerance: matched+=1;i+=1;j+=1
        elif d<0:i+=1
        else:j+=1
    return 2*matched/max(1,len(left)+len(right))


def evaluate_review_metrics(*, blueprint, symbolic_outputs=None, audio_manifest=None,
                            prompt="", reference_audio_path="", fad_reference_dir="",
                            imagebind_video_embedding="", imagebind_audio_embedding="",
                            inception_probabilities="", clap_model_dir="", clap_device=None,
                            reconstruction_reference_audio_path="", visual_features=None,
                            metric_inputs=None):
    from .paper_metrics import (_ok,_missing,_load_audio,_si_sdr,_clap_similarity)
    from .definitions import (emotion_alignment,color_timbre_alignment,chroma_diversity,
                              harmonic_consistency,dynamic_variation,waveform_features,segment_chroma)
    from reward.evidence import extract_audio_features
    spec=metric_inputs or {}
    names=("rhythm_control","emotion_alignment_score","color_timbre_alignment","multi_agent_music_proficiency",
           "dynamic_variation","frechet_audio_distance","inception_score","imagebind_score","si_sdr",
           "prompt_alignment","reference_style_similarity","harmonic_consistency","human_control_satisfaction","human_rating")
    metrics={k:_missing("explicit evaluator inputs/protocol are required",k) for k in names}
    diagnostics={}
    soundtrack=str((audio_manifest or {}).get("soundtrack_path", ""))
    audio_available=bool(soundtrack and Path(soundtrack).is_file())
    if audio_available:
        waveform,sr=_load_audio(soundtrack)
        feats=extract_audio_features(soundtrack)
        spectral=waveform_features(waveform,sr)
        chroma_cfg=spec.get("chroma",{})
        segment_seconds=float(chroma_cfg.get("segment_seconds",2.))
        bins=int(chroma_cfg.get("frames_per_segment",32))
        try:
            matrices=segment_chroma(spectral,duration=len(waveform)/sr,segment_seconds=segment_seconds,frames_per_segment=bins)
            metrics["chroma_diversity"]=_ok(chroma_diversity(matrices),"one_minus_pairwise_segment_chroma_cosine",
                segment_seconds=segment_seconds,frames_per_segment=bins,n_segments=len(matrices),
                trailing_partial_segment="discard",similarity="cosine_of_flattened_equal_shape_matrices",
                extraction=spectral["protocol"],interpretation="diversity only; not music quality")
        except ValueError as exc: metrics["chroma_diversity"]=_missing(str(exc),"one_minus_pairwise_segment_chroma_cosine")
        if spec.get("hcs"):
            import librosa
            hcs=spec["hcs"]
            if not hcs.get("scale_source"):raise ValueError("HCS requires the source of sigma_TC")
            try:
                if not spectral["active"].all():raise ValueError("HCS contains silent analysis frames; define a separate silence protocol")
                tc=librosa.feature.tonnetz(chroma=spectral["chroma"],sr=sr).T
                metrics["harmonic_consistency"]=_ok(harmonic_consistency(tc,sigma_tc=hcs["sigma_tc"]),
                    "tonal_centroid_step_distance",protocol=hcs,extraction=spectral["protocol"],
                    clipping="none",interpretation="smoothness only; constant harmony scores 1")
            except ValueError as exc: metrics["harmonic_consistency"]=_missing(str(exc),"tonal_centroid_step_distance")
        if spec.get("dynamic_variation"):
            from experiments.provenance import sha256_file
            dv=spec["dynamic_variation"]
            reference_path=dv["reference_audio"]
            if dv.get("reference_role")!="held_out_evaluation":raise ValueError("DV needs an explicit held_out_evaluation reference role")
            if reference_audio_path and sha256_file(reference_path)==sha256_file(reference_audio_path):
                raise ValueError("DV reference duplicates the conditioning reference; use a separate controlled reconstruction protocol")
            ref,ref_sr=_load_audio(reference_path)
            if ref_sr!=sr or len(ref)!=len(waveform):raise ValueError("DV requires equal sample rate/duration; align explicitly without time stretching")
            ref_features=waveform_features(ref,ref_sr)
            try:
                if not ref_features["active"].any() or not spectral["active"].any():raise ValueError("DV is undefined for entirely silent audio")
                metrics["dynamic_variation"]=_ok(dynamic_variation(ref_features["rms_db"],spectral["rms_db"]),
                    "one_minus_db_trajectory_cosine",reference_sha256=sha256_file(reference_path),reference_role=dv["reference_role"],
                    alignment="exact_start_no_search_no_time_warp",normalization="no_per_clip_peak_normalization",extraction=spectral["protocol"])
            except ValueError as exc:metrics["dynamic_variation"]=_missing(str(exc),"one_minus_db_trajectory_cosine")
        if visual_features:
            try:
                times=np.asarray(visual_features["times"],dtype=float)
                if len(times)<3 or (np.diff(times)<=0).any(): raise ValueError("visual timestamps must be increasing")
                source_hz=float(visual_features["sample_hz"])
                x=.65*np.asarray(visual_features["motion_intensity"])+.35*np.asarray(visual_features["cut_flags"])
                audio_onset=np.asarray(feats["onset"])
                audio_times=np.asarray(feats["frame_times"])
                y=np.interp(times,audio_times,audio_onset)
                value,lag=lagged_ncc(x,y,sample_hz=source_hz,max_lag_seconds=spec.get("max_lag_seconds",.1))
                metrics["rhythm_control"]=_ok(value,"visual_audio_max_lag_ncc",source_hz=source_hz,
                    max_lag_seconds=spec.get("max_lag_seconds",.1),best_lag_seconds=lag,
                    definition="max zero-mean NCC(0.65*motion+0.35*cut_flag, audio_onset); no circular shift",
                    onset_backend="numpy_positive_magnitude_spectral_flux_20ms_window_10ms_hop",
                    reconstruction_deviation="paper specifies madmom; this spectral-flux reconstruction requires a separate comparison protocol")
            except (ValueError,KeyError) as exc: metrics["rhythm_control"]=_missing(str(exc),"visual_audio_max_lag_ncc")
            try:
                times=np.asarray(visual_features["times"],float)
                mask=(times>=spectral["times"][0])&(times<=spectral["times"][-1])
                centroid=np.interp(times[mask],spectral["times"],spectral["centroid"])
                slope=np.interp(times[mask],spectral["times"],spectral["slope"])
                value=color_timbre_alignment(np.asarray(visual_features["brightness"])[mask],
                    np.asarray(visual_features["color_warmth"])[mask],centroid,slope)
                metrics["color_timbre_alignment"]=_ok(value,"half_sum_brightness_centroid_warmth_negative_slope_correlations",
                    extraction=spectral["protocol"],alignment="interpolate_audio_to_measured_visual_times_within_overlap")
            except (ValueError,KeyError) as exc: metrics["color_timbre_alignment"]=_missing(str(exc),"color_timbre_correlation")
        else:
            metrics["rhythm_control"]=_missing("measured visual event curves required; pooled segment values do not identify event timing","visual_audio_max_lag_ncc")
        rms=np.sqrt(np.mean(np.square(waveform.reshape(-1))))
        diagnostics["rms_dbfs"]=_ok(20*np.log10(max(float(rms),1e-6)),"full_scale_1.0",silence_floor_db=-120)
        events=[e for item in symbolic_outputs or [] for e in item.get("events",[])]
        if events:
            from scipy.signal import find_peaks
            onset=np.asarray(feats["onset"]); hz=feats["analysis_hz"]
            peaks,_=find_peaks(onset,height=.2,distance=max(1,int(.05*hz)))
            diagnostics["score_audio_onset_f1"]=_ok(onset_f1({e["onset"] for e in events},feats["frame_times"][peaks]),"waveform_peak_vs_score_event",tolerance_seconds=.05)
        if reconstruction_reference_audio_path:
            reference,ref_sr=_load_audio(reconstruction_reference_audio_path)
            if abs(len(reference)/ref_sr-len(waveform)/sr)>1/max(ref_sr,sr):
                metrics["si_sdr"]=_missing("reconstruction reference duration differs; align explicitly before evaluation","scale_invariant_sdr")
            elif np.std(reference)<1e-10:
                metrics["si_sdr"]=_missing("silent reference makes SI-SDR undefined","scale_invariant_sdr")
            else:
                metrics["si_sdr"]=_ok(_si_sdr(soundtrack,reconstruction_reference_audio_path),"scale_invariant_sdr",unit="dB",
                    task="explicit_reference_reconstruction_only",sample_rate=min(sr,ref_sr),channel_policy="mono_average",alignment="start_aligned_no_search")
        else:
            metrics["si_sdr"]=_missing("disabled for free generation; supply a separate reconstruction reference","scale_invariant_sdr")
        if prompt and clap_model_dir:
            metrics["prompt_alignment"]=_ok(_clap_similarity(prompt,soundtrack,clap_model_dir,clap_device),"clap_affine_cosine",checkpoint=str(clap_model_dir),transform="(cosine+1)/2")
    else:
        metrics["chroma_diversity"]=_missing("generated waveform required","spectral_chroma")
    # Models, sampling and candidate sets are supplied explicitly rather than inferred from file names.
    if imagebind_video_embedding and imagebind_audio_embedding and spec.get("imagebind_checkpoint"):
        v,a=np.load(imagebind_video_embedding),np.load(imagebind_audio_embedding)
        candidates=np.load(spec["imagebind_candidates"]) if spec.get("imagebind_candidates") else None
        mode="matched_candidate_softmax" if candidates is not None else "raw_cosine"
        metrics["imagebind_score"]=_ok(imagebind_score(v,a,candidates=candidates,target_index=spec.get("imagebind_target_index"),temperature=spec.get("imagebind_temperature")),
            "precomputed_imagebind",mode=mode,checkpoint=spec["imagebind_checkpoint"],candidate_ids=spec.get("imagebind_candidate_ids"),temperature=spec.get("imagebind_temperature"))
        if candidates is not None and len(spec.get("imagebind_candidate_ids",[]))!=len(candidates):
            raise ValueError("candidate IDs must be recorded for every ImageBind candidate")
    if spec.get("fad_generated_embeddings") and spec.get("fad_reference_embeddings"):
        required=("fad_checkpoint","fad_reference_split","fad_window_seconds","fad_hop_seconds","fad_generated_ids","fad_reference_ids")
        if any(k not in spec for k in required): raise ValueError("FAD protocol missing checkpoint/split/window/sample IDs")
        g,r=np.load(spec["fad_generated_embeddings"]),np.load(spec["fad_reference_embeddings"])
        if len(spec["fad_generated_ids"])!=len(g) or len(spec["fad_reference_ids"])!=len(r): raise ValueError("FAD IDs do not match embedding rows")
        metrics["frechet_audio_distance"]=_ok(frechet_embeddings(g,r),"precomputed_audio_embeddings",n_generated=len(g),n_reference=len(r),protocol={k:spec[k] for k in required})
    if inception_probabilities and spec.get("inception_checkpoint"):
        metrics["inception_score"]=_ok(inception_score(np.load(inception_probabilities)),"categorical_classifier",checkpoint=spec["inception_checkpoint"])
    for name,key in (("emotion_alignment_score","emotion"),("reference_style_similarity","style")):
        config=spec.get(key)
        if config:
            if not config.get("checkpoint") or not config.get("protocol"): raise ValueError(f"{key} needs checkpoint and protocol")
            left,right=np.load(config["generated"]),np.load(config["reference"])
            if left.shape!=right.shape or left.ndim!=2 or not np.isfinite(left).all() or not np.isfinite(right).all(): raise ValueError("evaluator features must align exactly")
            if key=="emotion":
                if left.shape[1]!=2: raise ValueError("VA arrays must be [T,2]")
                value,lag=emotion_alignment(right,left,sample_hz=config["sample_hz"],max_lag_seconds=config["max_lag_seconds"])
                config={**config,"best_lag_seconds":lag,"definition":"max_shared_lag_mean_spearman_VA; average ranks per overlap; positive lag means audio delayed"}
            else:
                value=float(np.mean([imagebind_score(a,b) for a,b in zip(left,right)]))
            metrics[name]=_ok(value,"explicit_precomputed_"+key,protocol=config)
    if spec.get("uniform_transcription_report") and soundtrack:
        import json
        from .uniform_transcription import select_measurement
        report=json.loads(Path(spec["uniform_transcription_report"]).read_text())
        value,protocol_id=select_measurement(report,soundtrack)
        metrics["multi_agent_music_proficiency"]=_ok(value,"uniform_audio_transcription_v1",protocol_id=protocol_id,
            protocol=report["protocol"],source_report=spec["uniform_transcription_report"])
    aliases={"RC":"rhythm_control","EAS":"emotion_alignment_score","CTA":"color_timbre_alignment","ImageBind":"imagebind_score",
             "MMP":"multi_agent_music_proficiency","DV":"dynamic_variation","FAD":"frechet_audio_distance","IS":"inception_score",
             "SI-SDR":"si_sdr","PAlign":"prompt_alignment","RStyle":"reference_style_similarity","HCS":"harmonic_consistency","Human":"human_rating"}
    metrics["metric_aliases"]={a:metrics[k] for a,k in aliases.items()}
    return {"protocol_version":VERSION,"metrics":metrics,"diagnostics":diagnostics,"conditioning_reference":reference_audio_path,
            "reconstruction_reference":reconstruction_reference_audio_path,"generated_audio_paths":(audio_manifest or {}).get("segment_paths",[]),
            "notes":{"unavailable":"Missing model/protocol/data is not a zero score.","resolution":"No interpolated curve is labelled as new visual evidence.",
                     "HCS":"Harmonic consistency, not human control satisfaction.","historical_numbers":"v2 scores cannot replace paper numbers without rerunning all systems."}}
