"""Paired, clustered uncertainty. Seeds and repeated ratings are not independent videos."""
import numpy as np


def paired_video_bootstrap(left,right,*,replicates=5000,seed=123):
    # Mapping keys are (video_id, seed); all pairs must match, never silently intersect.
    if set(left)!=set(right) or not left:raise ValueError("paired sample/seed keys must match exactly")
    videos=sorted({key[0] for key in left})
    if len(videos)<2:raise ValueError("at least two independent videos are required")
    differences=np.asarray([np.mean([left[k]-right[k] for k in left if k[0]==v]) for v in videos])
    if not np.isfinite(differences).all():raise ValueError("nonfinite paired result")
    rng=np.random.default_rng(seed)
    estimates=np.asarray([rng.choice(differences,size=len(videos),replace=True).mean() for _ in range(replicates)])
    return {"mean_difference":float(differences.mean()),"ci95":np.quantile(estimates,[.025,.975]).tolist(),
            "n_independent_videos":len(videos),"pair_count":len(left),"unit":"video (seeds averaged within video)","replicates":replicates,"seed":seed}


def crossed_human_bootstrap(rows,method_a,method_b,*,replicates=5000,seed=123):
    cells={}
    for row in rows:
        key=(row["participant_id"],row["video_id"],row["dimension"],row["method"])
        if key in cells:raise ValueError("duplicate participant/video/dimension/method rating")
        cells[key]=float(row["rating"])
    dims={r["dimension"] for r in rows}
    if len(dims)!=1:raise ValueError("analyze one rating dimension at a time")
    d=next(iter(dims));a={(p,v):x for (p,v,k,m),x in cells.items() if m==method_a};b={(p,v):x for (p,v,k,m),x in cells.items() if m==method_b}
    if set(a)!=set(b) or not a:raise ValueError("human methods require paired participant/video cells")
    participants=sorted({p for p,v in a});videos=sorted({v for p,v in a})
    if min(len(participants),len(videos))<2:raise ValueError("need >=2 participants and >=2 independent videos")
    differences=np.asarray([a[k]-b[k] for k in a]);keys=list(a)
    if not np.isfinite(differences).all():raise ValueError("nonfinite human rating")
    pi={p:i for i,p in enumerate(participants)};vi={v:i for i,v in enumerate(videos)}
    rng=np.random.default_rng(seed);means=[]
    for _ in range(replicates):
        pc=np.bincount(rng.integers(len(participants),size=len(participants)),minlength=len(participants))
        vc=np.bincount(rng.integers(len(videos),size=len(videos)),minlength=len(videos))
        weights=np.asarray([pc[pi[p]]*vc[vi[v]] for p,v in keys])
        if weights.sum():means.append(float(np.average(differences,weights=weights)))
    return {"mean_difference":float(differences.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),"participants":len(participants),"videos":len(videos),"paired_cells":len(keys),"dimension":d,"bootstrap":"two-way participant x video (pigeonhole)","replicates_used":len(means)}


def holm_correction(pvalues):
    values=np.asarray(pvalues,dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError("invalid p-values")
    order=np.argsort(values);adjusted=np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values))))
    result=np.empty_like(values);result[order]=np.minimum(1,adjusted);return result.tolist()


def paired_video_signflip(left,right,*,replicates=10000,seed=123):
    """Two-sided paired randomization test; seed replicates stay inside videos."""
    if not left or set(left)!=set(right):raise ValueError("paired keys must match exactly")
    videos=sorted({key[0] for key in left})
    if len(videos)<2:raise ValueError("at least two video units required")
    diffs=np.array([np.mean([left[k]-right[k] for k in left if k[0]==v]) for v in videos])
    if not np.isfinite(diffs).all():raise ValueError("nonfinite paired differences")
    observed=abs(diffs.mean())
    if len(videos)<=12:
        import itertools
        statistics=np.array([abs(np.mean(diffs*np.array(signs))) for signs in itertools.product([-1,1],repeat=len(videos))])
        p=float(np.mean(statistics>=observed-1e-15));method="exact sign flip"
    else:
        if replicates<1:raise ValueError("positive replicate count required")
        rng=np.random.default_rng(seed)
        statistics=np.abs((rng.choice([-1,1],size=(replicates,len(videos)))*diffs).mean(axis=1))
        p=float((1+np.sum(statistics>=observed-1e-15))/(replicates+1));method="Monte Carlo sign flip, plus-one correction"
    return {"mean_difference":float(diffs.mean()),"p_value":p,"method":method,"n_videos":len(videos),
            "per_video_differences":dict(zip(videos,diffs.tolist())),"randomizations":len(statistics),
            "assumption":"paired label exchangeability within independent video units; seeds averaged before testing"}
