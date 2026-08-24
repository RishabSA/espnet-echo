import argparse
import json
from pathlib import Path

from scripts.common.io import write_jsonl

skip_keys = {"docs", "argv"}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k in skip_keys:
            continue
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif not isinstance(v, list):
            out[f"{prefix}{k}"] = v
    return out


def aggregate(root: str | Path) -> list[dict]:
    rows = []
    for path in sorted(Path(root).glob("**/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"summary_path": str(path), **flatten(summary)})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect every summary.json under a runs dir into one flat JSONL, one row per evaluation, for cross-run queries.")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory searched recursively for metrics/*/summary.json (default: runs).")
    parser.add_argument("--out", type=str, default="runs/_tables/runs.jsonl", help="Output JSONL (default: runs/_tables/runs.jsonl).")
    args = parser.parse_args()

    rows = aggregate(args.runs_dir)
    write_jsonl(args.out, rows)
    for r in rows:
        print(f"{r['run_name']:28s} {r['phase']}_{r['split']:5s} wer={r['metrics.wer']:.4f} ccr={r['metrics.consistency.ccr']:.4f}")
    print(f"{len(rows)} evaluations -> {args.out}")
