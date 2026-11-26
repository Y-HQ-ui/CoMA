"""Audit explicit train/validation/test manifests and conditioning provenance."""
import argparse
import json
from pathlib import Path


def audit_dataset(rows):
    errors=[];seen=set();owners={};counts={};references=[]
    for row in rows:
        sid=row.get("sample_id");split=row.get("split")
        if not sid or sid in seen:errors.append(f"missing/duplicate sample_id: {sid}")
        seen.add(sid)
        if split not in {"train","validation","test"}:errors.append(f"{sid}: invalid split")
        counts[split]=counts.get(split,0)+1
        if not row.get("dataset") or not row.get("movie_id") or not row.get("music_work_id"):
            errors.append(f"{sid}: dataset/movie_id/music_work_id are mandatory for leakage audit")
        # IDs must be globally canonical across source datasets, not per-dataset prefixes.
        for kind in ("movie_id","music_work_id","video_sha256","target_audio_sha256"):
            key=(kind,row.get(kind))
            if not key[1]:continue
            if key in owners and owners[key]!=split:errors.append(f"{sid}: {kind} {key[1]} crosses {owners[key]}/{split}")
            owners[key]=split
        condition=row.get("input_condition")
        if condition not in {"V","V+q","V+r","V+q+r"}:errors.append(f"{sid}: invalid condition")
        if "q" in str(condition) and (not row.get("prompt_source") or row.get("prompt_used_target_audio") is not False):
            errors.append(f"{sid}: q provenance must explicitly exclude target audio access")
        if "r" in str(condition):
            required=("reference_audio_id","reference_music_work_id","reference_movie_id","reference_source")
            if any(not row.get(k) for k in required):errors.append(f"{sid}: missing reference provenance")
            if row.get("reference_audio_id")==row.get("target_audio_id") or row.get("reference_music_work_id")==row.get("music_work_id") or row.get("reference_movie_id")==row.get("movie_id"):
                errors.append(f"{sid}: reference reuses target recording/work/movie")
            if row.get("reference_audio_sha256") and row["reference_audio_sha256"]==row.get("target_audio_sha256"):
                errors.append(f"{sid}: reference equals target waveform")
            references.append(row)
    for row in references:
        for ref_key,target_key in (("reference_music_work_id","music_work_id"),("reference_movie_id","movie_id")):
            owner=owners.get((target_key,row.get(ref_key)))
            if owner and owner!=row["split"]:errors.append(f"{row['sample_id']}: reference {ref_key} belongs to {owner} split")
    return {"valid":not errors,"errors":sorted(set(errors)),"counts":counts,"sample_count":len(rows),
            "limitation":"metadata/hash audit; unknown pretraining overlap and perceptual near-duplicates require separate identification"}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest");p.add_argument("--output")
    args=p.parse_args();rows=[json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    result=audit_dataset(rows);content=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(content)
    print(content);return 0 if result["valid"] else 1

if __name__=="__main__":raise SystemExit(main())
