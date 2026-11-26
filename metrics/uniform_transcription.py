"""Uniform waveform-to-ABC-to-evaluator protocol for MMP-style measurements.

The manuscript does not identify the transcriber, evaluator, or prompt. These
must be supplied explicitly; clean CoMA scores are never accepted as shortcuts.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
from experiments.provenance import sha256_file, canonical_hash


def evaluate_records(records, protocol, transcribe, evaluate):
    required = {"transcriber_checkpoint", "evaluator_checkpoint", "prompt", "decoding", "score_definition"}
    if not required <= set(protocol) or any(not protocol[k] for k in required):
        raise ValueError("explicit transcriber/evaluator/prompt/decoding/score definition required")
    if not records:
        raise ValueError("no audio records")
    keys, output = set(), []
    for row in records:
        key = row["sample_id"], row["method"], row["seed"]
        if key in keys:
            raise ValueError("duplicate MMP evaluation record")
        keys.add(key)
        audio = Path(row["audio"])
        if "abc" in row or "symbolic_outputs" in row:
            raise ValueError("uniform evaluation must transcribe every waveform; clean symbolic bypass is forbidden")
        item = {**row, "audio_sha256": sha256_file(audio)}
        try:
            transcription = transcribe(audio_path=str(audio), checkpoint=protocol["transcriber_checkpoint"],
                                       decoding=protocol["decoding"])
            abc = transcription["abc"]
            if transcription.get("checkpoint") != protocol["transcriber_checkpoint"] or not isinstance(abc, str) or not abc.strip():
                raise ValueError("transcriber version/output mismatch")
            result = evaluate(abc=abc, checkpoint=protocol["evaluator_checkpoint"], prompt=protocol["prompt"],
                              decoding=protocol["decoding"], score_definition=protocol["score_definition"])
            score = float(result["score"])
            if result.get("checkpoint") != protocol["evaluator_checkpoint"] or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("evaluator version/score mismatch; explicit [0,1] score required")
            item.update(status="ok", value=score, transcription=transcription, evaluator=result)
        except (ValueError, KeyError, RuntimeError) as exc:
            item.update(status="failed", error=str(exc))
        output.append(item)
    return {"version": "uniform_audio_transcription_v1", "protocol": protocol,
            "protocol_id": canonical_hash(protocol), "records": output,
            "policy": "all systems transcribed with the same checkpoint/prompt; failures retained; no native score bypass"}


def select_measurement(report, audio_path):
    digest = sha256_file(audio_path)
    if report.get("version") != "uniform_audio_transcription_v1" or report.get("protocol_id") != canonical_hash(report["protocol"]):
        raise ValueError("invalid uniform transcription protocol")
    matches = [row for row in report["records"] if row["audio_sha256"] == digest]
    if len(matches) != 1 or matches[0]["status"] != "ok":
        raise ValueError("need one successful MMP record bound to this exact waveform")
    return matches[0]["value"], report["protocol_id"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True);p.add_argument("--protocol", required=True)
    p.add_argument("--transcriber", required=True, help="module:function")
    p.add_argument("--evaluator", required=True, help="module:function")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    def plugin(name):
        module, function = name.split(":")
        return getattr(importlib.import_module(module), function)
    path = Path(args.manifest).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in records:
        audio = Path(row["audio"])
        row["audio"] = str(audio if audio.is_absolute() else path.parent/audio)
    protocol = json.loads(Path(args.protocol).read_text())
    protocol.update(transcriber_plugin=args.transcriber, evaluator_plugin=args.evaluator)
    result = evaluate_records(records, protocol, plugin(args.transcriber), plugin(args.evaluator))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "ok" for row in result["records"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
