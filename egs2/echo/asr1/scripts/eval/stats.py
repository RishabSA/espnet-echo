import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from scripts.common.io import read_jsonl

# every metric is a ratio of per-document sums; (numerator, denominator) keys of per_doc.jsonl
ratios = {
    "wer": ("wer_errors", "wer_n_ref"),
    "ecr_strict": ("consistent_strict", "eligible_entities"),
    "ecr_norm": ("consistent_norm", "eligible_entities"),
    "ecr_pairwise": ("pairwise_agreement", "eligible_entities"),
    "ccr": ("consistent_correct", "eligible_entities"),
    "entity_wer": ("entity_errors", "entity_ref_words"),
    "oracle_entity_wer": ("oracle_entity_errors", "entity_ref_words"),
    "occ_correct": ("eligible_correct", "eligible_realized"),
    "oracle_occ_correct": ("oracle_correct", "eligible_realized"),
    "retrieval_recall": ("retrieved_mentions", "ref_mentions"),
}
bias_ratios = {"b_wer": ("b_errors", "b_n_ref"), "u_wer": ("u_errors", "u_n_ref"), "baer": ("in_list_wrong", "in_list")}


def columns(per_doc: list[dict], num: str, den: str, group: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows = [d[group] if group else d for d in per_doc]
    # a document with nothing to count (no eligible entities, no biased words) carries no key; that is a zero
    return np.array([r.get(num, 0) for r in rows], dtype=float), np.array([r.get(den, 0) for r in rows], dtype=float)


def bootstrap(num: np.ndarray, den: np.ndarray, n_boot: int, seed: int, num2: np.ndarray | None = None, den2: np.ndarray | None = None) -> dict:
    # document-level resampling, ratio of sums per resample (never mean of ratios); a paired
    # comparison resamples the same documents for both runs and takes the difference
    if den.sum() == 0 or (den2 is not None and den2.sum() == 0):
        return {"point": None, "ci95": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))  # shape: (n_boot, n_docs)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = num[idx].sum(axis=1) / den[idx].sum(axis=1)  # shape: (n_boot,)
        point = num.sum() / den.sum()
        if num2 is not None:
            est = est - num2[idx].sum(axis=1) / den2[idx].sum(axis=1)
            point = point - num2.sum() / den2.sum()
    # a resample made only of documents with nothing to count has no ratio and drops out
    est = est[np.isfinite(est)]
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)]}


def paired_wilcoxon(num: np.ndarray, den: np.ndarray, num2: np.ndarray, den2: np.ndarray) -> float | None:
    with np.errstate(invalid="ignore", divide="ignore"):
        deltas = num / den - num2 / den2
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return None
    if np.all(deltas == 0):
        return 1.0
    return float(wilcoxon(deltas).pvalue)


def run_stats(per_doc: list[dict], n_boot: int, seed: int, compare: list[dict] | None = None) -> dict:
    metrics = dict(ratios)
    if "bias" in per_doc[0]:
        metrics.update({k: v for k, v in bias_ratios.items()})
    out = {"n_docs": len(per_doc), "n_boot": n_boot, "seed": seed, "metrics": {}}
    if compare is not None:
        if [d["doc_id"] for d in compare] != [d["doc_id"] for d in per_doc]:
            raise ValueError("paired stats need the same documents in the same order in both per_doc files")
        out["compare_n_docs"] = len(compare)
    for name, (num_key, den_key) in metrics.items():
        group = "bias" if name in bias_ratios else None
        num, den = columns(per_doc, num_key, den_key, group)
        entry = bootstrap(num, den, n_boot, seed)
        if compare is not None:
            num2, den2 = columns(compare, num_key, den_key, group)
            entry["diff"] = bootstrap(num, den, n_boot, seed, num2, den2)
            entry["diff"]["wilcoxon_p"] = paired_wilcoxon(num, den, num2, den2)
        out["metrics"][name] = entry
    return out


def merge_into_summary(summary_path: Path, result: dict, compare_name: str | None) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ci"] = {k: v["ci95"] for k, v in result["metrics"].items()}
    if compare_name is not None:
        summary["deltas_vs"] = {"run": compare_name, **{k: dict(v["diff"]) for k, v in result["metrics"].items()}}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document-level bootstrap CIs (ratio of sums) over a metrics dir's per_doc.jsonl, plus paired differences and Wilcoxon against a second run (spec 07 section 7.4).")
    parser.add_argument("--metrics-dir", type=str, required=True, help="Metrics dir written by evaluate.py, containing per_doc.jsonl (required).")
    parser.add_argument("--compare-metrics-dir", type=str, default=None, help="Second metrics dir for paired differences, this run minus that one (default: None).")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of bootstrap resamples (default: 10000).")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed (default: 42).")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: <metrics-dir>/stats.json, or stats_paired.json with --compare-metrics-dir).")
    args = parser.parse_args()

    per_doc = read_jsonl(Path(args.metrics_dir) / "per_doc.jsonl")
    compare = read_jsonl(Path(args.compare_metrics_dir) / "per_doc.jsonl") if args.compare_metrics_dir else None
    result = run_stats(per_doc, args.bootstrap, args.seed, compare)
    result["argv"] = vars(args)

    out = Path(args.out) if args.out else Path(args.metrics_dir) / ("stats_paired.json" if compare else "stats.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_path = Path(args.metrics_dir) / "summary.json"
    if summary_path.exists():
        compare_name = None
        if compare is not None:
            cs = Path(args.compare_metrics_dir) / "summary.json"
            compare_name = f"{json.loads(cs.read_text())['run_name']}/{Path(args.compare_metrics_dir).name}" if cs.exists() else args.compare_metrics_dir
        merge_into_summary(summary_path, result, compare_name)
    for name, e in result["metrics"].items():
        line = f"{name:20s} {e['point']:.4f}  [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}]"
        if "diff" in e:
            d = e["diff"]
            line += f"  diff {d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] p={d['wilcoxon_p']:.3f}"
        print(line)
