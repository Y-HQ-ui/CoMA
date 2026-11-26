"""Export table and radar from one validated data file, preserving metric order."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .provenance import canonical_hash

COMPARISON_FIELDS=("split_id","sample_ids","input_condition","seeds","max_generation_calls","evaluation_version","evaluation_settings","prompt_source","reference_source","duration_seconds","aggregation")


def aggregate_runs(runs,metric_names):
    if not runs:raise ValueError("no runs")
    comparison={k:runs[0]["protocol"][k] for k in COMPARISON_FIELDS}
    rows=[];names=set()
    for run in runs:
        if {k:run["protocol"][k] for k in COMPARISON_FIELDS}!=comparison:raise ValueError("incompatible table protocols; export separate tables")
        method=run["method"]
        if method in names:raise ValueError("duplicate method")
        names.add(method)
        expected={(s,seed) for s in comparison["sample_ids"] for seed in comparison["seeds"]}
        keys=[(r["sample_id"],r["seed"]) for r in run["records"]]
        if set(keys)!=expected or len(keys)!=len(expected):raise ValueError("incomplete/duplicate sample records")
        if any(r.get("status")!="ok" for r in run["records"]):raise ValueError("failed samples cannot be dropped from a paper table; define an explicit failure protocol")
        row={"method":method,"n_videos":len(comparison["sample_ids"]),"n_seeds":len(comparison["seeds"]),"metrics":{}}
        for metric in metric_names:
            entries=[r["metrics"][metric] for r in run["records"]]
            if any(e.get("status")!="ok" for e in entries):raise ValueError(f"unavailable {metric} in {method}")
            backends={e["backend"] for e in entries}
            if len(backends)!=1:raise ValueError("mixed metric backends")
            vals=np.asarray([e["value"] for e in entries],dtype=float)
            if not np.isfinite(vals).all():raise ValueError("nonfinite result")
            row["metrics"][metric]={"mean":float(vals.mean()),"std_across_sample_seed":float(vals.std(ddof=1)) if len(vals)>1 else None,"backend":next(iter(backends))}
            seed_means={str(seed):float(np.mean([r["metrics"][metric]["value"] for r in run["records"] if r["seed"]==seed])) for seed in comparison["seeds"]}
            row["metrics"][metric].update(per_seed_means=seed_means,
                std_of_seed_means=float(np.std(list(seed_means.values()),ddof=1)) if len(seed_means)>1 else None,
                uncertainty_note="pooled sample-seed standard deviation is descriptive; paired inference clusters by video")
        rows.append(row)
    for metric in metric_names:
        if len({r["metrics"][metric]["backend"] for r in rows})!=1:raise ValueError("methods use incompatible evaluators")
    return {"comparison":comparison,"rows":rows,"source_ids":[r.get("protocol_id",canonical_hash(r["protocol"])) for r in runs]}


def normalized_radar(table,directions):
    result={r["method"]:[] for r in table["rows"]};ranges={}
    for metric,direction in directions.items():
        if direction not in {"higher","lower"}:raise ValueError("direction must be higher/lower")
        values=np.asarray([r["metrics"][metric]["mean"] for r in table["rows"]]);lo,hi=values.min(),values.max()
        scaled=(values-lo)/(hi-lo) if hi>lo else np.full(len(values),.5)
        if direction=="lower":scaled=1-scaled
        ranges[metric]={"min":float(lo),"max":float(hi),"direction":direction}
        for row,value in zip(table["rows"],scaled):result[row["method"]].append(float(value))
    return {"values":result,"ranges":ranges,"formula":"(x-min)/(max-min), reversed for lower-is-better; tied axis=0.5"}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("runs",nargs="+");p.add_argument("--directions",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--radar",action="store_true")
    args=p.parse_args();directions=json.loads(Path(args.directions).read_text());table=aggregate_runs([json.loads(Path(x).read_text()) for x in args.runs],list(directions));radar=normalized_radar(table,directions)
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    (root/"table_data.json").write_text(json.dumps({**table,"normalization":radar},indent=2))
    with (root/"table.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["method",*directions])
        for row in table["rows"]:writer.writerow([row["method"],*[row["metrics"][m]["mean"] for m in directions]])
    if args.radar:
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        angles=np.linspace(0,2*np.pi,len(directions),endpoint=False);angles=np.r_[angles,angles[0]]
        fig,ax=plt.subplots(subplot_kw={"projection":"polar"})
        for method,values in radar["values"].items():ax.plot(angles,[*values,values[0]],label=method)
        ax.set_xticks(angles[:-1],directions);ax.set_ylim(0,1);ax.legend(bbox_to_anchor=(1.1,1));fig.savefig(root/"radar.png",bbox_inches="tight",dpi=160);plt.close(fig)
    return 0

if __name__=="__main__":raise SystemExit(main())
