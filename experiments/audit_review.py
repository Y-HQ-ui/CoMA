"""Check paper/review coverage links; never certify experimental completion."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from experiments.provenance import sha256_file, canonical_hash


def audit(matrix_path, source_root):
    path, root = Path(matrix_path).resolve(), Path(source_root).resolve()
    matrix = json.loads(path.read_text())
    statuses = matrix["status_definitions"]
    expected_statuses = {"implemented", "reconstructed", "partial", "requires_original_evidence"}
    if set(statuses) != expected_statuses:
        raise ValueError("matrix status definitions changed or are incomplete")
    items = matrix["items"]
    if not items or len({item["id"] for item in items}) != len(items):
        raise ValueError("empty matrix or duplicate requirement ID")
    groups = {int(match.group(1)) for item in items if (match := re.match(r"R(\d+)\.", item["id"]))}
    if groups != set(range(1, 12)):
        raise ValueError("all eleven review issue groups must be covered")
    if not {f"P{n}" for n in range(10, 19)} <= {item["id"] for item in items}:
        raise ValueError("explicit reward equation entries P10--P18 required")
    links = dict(matrix["sources"])
    for item in items:
        if item["implementation_status"] not in statuses or not item["requirement"] or not item["source"]:
            raise ValueError(f"invalid requirement: {item['id']}")
        if item["implementation_status"] != "implemented" and not item["remaining"]:
            raise ValueError(f"unfinished requirement must explain its limitation: {item['id']}")
        if not item["implementation"]:
            raise ValueError(f"requirement needs a concrete evidence or explanation path: {item['id']}")
        if item["implementation_status"] == "implemented" and not item["verification"]:
            raise ValueError(f"implemented requirement needs a verification link: {item['id']}")
        for relative in [*item["implementation"], *item["verification"]]:
            links[relative] = relative
    digests = {}
    for relative in sorted(set(links.values())):
        target = (root/relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"missing or out-of-workspace evidence path: {relative}")
        digests[relative] = sha256_file(target)
    return {"status": "coverage_links_valid", "requirements": len(items),
            "review_groups": sorted(groups), "counts": dict(Counter(i["implementation_status"] for i in items)),
            "matrix_sha256": sha256_file(path), "evidence_files": digests,
            "evidence_fingerprint": canonical_hash(digests),
            "scope": "schema, eleven-group coverage and file existence/hash only; does not run tests or certify scientific completion"}


def markdown(matrix):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    lines = ["# 论文与审稿意见逐项实现表", "", matrix["scope"], "",
             "本表由用户提供的实现矩阵生成。实现状态指代码覆盖范围；有代码和测试不代表正式实验已完成。", "",
             "## 状态含义", ""]
    lines.extend(f"- `{key}`：{value}" for key, value in matrix["status_definitions"].items())
    lines += ["", "## 逐条对应", "", "| ID / 来源 | 要求 | 状态 | 实现入口 | 验证入口 | 剩余事项 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for item in matrix["items"]:
        row = [item["id"]+" / "+item["source"], item["requirement"], item["implementation_status"],
               "<br>".join(f"`{p}`" for p in item["implementation"]),
               "<br>".join(f"`{p}`" for p in item["verification"]) or "无自动验证；见剩余事项",
               item["remaining"] or "本条代码已实现；正式研究结论另需实验依据"]
        lines.append("| "+" | ".join(map(cell, row))+" |")
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True)
    p.add_argument("--source-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--markdown")
    args = p.parse_args()
    result = audit(args.matrix, args.source_root)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("audit history is immutable; choose a new output file")
    if args.markdown:
        rendered = markdown(json.loads(Path(args.matrix).read_text()))
        Path(args.markdown).write_text(rendered)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("status", "requirements", "counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
