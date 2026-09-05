import argparse
import json
import os
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scripts.common.align import align_doc
from scripts.common.io import append_config, git_sha, read_hyp_text, read_jsonl
from scripts.eval.evaluate import alignable_entities
from scripts.mine.cluster_variants import build_clusters, distance_matrices
from scripts.mine.eval_clusters import (
    doc_cluster_metrics,
    hyp_word_to_ref,
    mention_labels,
    ref_word_entities,
    summarize,
)

percentiles = (50, 90, 95, 99)
table_cols = ["purity", "contamination", "vote_damage_rate", "vote_repair_rate", "oracle_damage_rate", "oracle_repair_rate", "coverage_occ", "list_precision", "list_recall", "list_recall_any_variant", "clusters_per_doc", "named_cluster_share", "named_median_n_occ", "named_share_n_occ_2"]


def floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def labeled_pairs(prepared: list[dict]) -> dict:
    # every pair of named-entity variants inside a document, split by whether their reference
    # spellings agree (the label a cluster must not mix)
    same, diff = {"lev": [], "phon": []}, {"lev": [], "phon": []}
    for d in prepared:
        votes = {}
        for m in d["cands"]:
            spelling, _, is_named = mention_labels(m, d["alignment"], d["word_ref"], d["by_word"], d["named"])
            if spelling is not None and is_named:
                votes.setdefault(m["norm"], Counter())[spelling] += 1
        idx = {v: i for i, v in enumerate(d["variants"])}
        lev, phon, _ = d["mats"]
        labeled = [(v, c.most_common(1)[0][0]) for v, c in votes.items()]
        for (va, la), (vb, lb) in combinations(labeled, 2):
            bucket = same if la == lb else diff
            bucket["lev"].append(lev[idx[va], idx[vb]])
            bucket["phon"].append(phon[idx[va], idx[vb]])
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in (("same_entity", same), ("different_entity", diff))}


def pair_report(pairs: dict, lambdas: list[float], taus: list[float], caps: list[float]) -> dict:
    def q(xs: np.ndarray) -> dict | None:
        return {str(p): float(np.percentile(xs, p)) for p in percentiles} if len(xs) else None

    out = {k: {"n": len(v["lev"]), "lev": q(v["lev"]), "phon": q(v["phon"])} for k, v in pairs.items()}
    same, diff = pairs["same_entity"], pairs["different_entity"]
    out["cap"] = [{"dphon_cap": c, "same_blocked": float(np.mean(same["phon"] > c)) if len(same["phon"]) else None,
                   "different_blocked": float(np.mean(diff["phon"] > c)) if len(diff["phon"]) else None} for c in caps]
    out["threshold"] = []
    for lam, tau in product(lambdas, taus):
        ds, dd = lam * same["lev"] + (1 - lam) * same["phon"], lam * diff["lev"] + (1 - lam) * diff["phon"]
        out["threshold"].append({"lambda": lam, "tau": tau, "same_within": float(np.mean(ds <= tau)) if len(ds) else None,
                                 "different_within": float(np.mean(dd <= tau)) if len(dd) else None})
    return out


def fmt(v: float | None) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render(rows: list[dict], pairs: dict) -> str:
    lines = ["| lambda | cap | tau | " + " | ".join(table_cols) + " |", "|" + "---|" * (len(table_cols) + 3)]
    lines += ["| " + " | ".join([fmt(r["lambda"]), fmt(r["dphon_cap"]), fmt(r["tau"])] + [fmt(r[c]) for c in table_cols]) + " |" for r in rows]
    recall_line = (f"Candidate recall (before clustering): occ {fmt(rows[0]['candidate_recall_occ'])}, entity {fmt(rows[0]['candidate_recall_entity'])}; "
                   f"{fmt(rows[0]['candidates_per_doc'])} candidates and {fmt(rows[0]['candidate_norms_per_doc'])} distinct spellings per doc.")
    lines += ["", recall_line, "", "Pairwise distances between named-entity variants (same reference spelling vs different):", "",
              "| pairs | n | lev p50/p90/p95/p99 | phon p50/p90/p95/p99 |", "|---|---|---|---|"]
    for k in ("same_entity", "different_entity"):
        p = pairs[k]
        lines.append(f"| {k} | {p['n']} | " + "/".join(fmt(p["lev"][str(x)]) for x in percentiles) + " | " + "/".join(fmt(p["phon"][str(x)]) for x in percentiles) + " |")
    lines += ["", "| cap | same-entity pairs blocked | different-entity pairs blocked |", "|---|---|---|"]
    lines += [f"| {fmt(c['dphon_cap'])} | {fmt(c['same_blocked'])} | {fmt(c['different_blocked'])} |" for c in pairs["cap"]]
    lines += ["", "| lambda | tau | same-entity pairs within tau | different-entity pairs within tau |", "|---|---|---|---|"]
    lines += [f"| {fmt(t['lambda'])} | {fmt(t['tau'])} | {fmt(t['same_within'])} | {fmt(t['different_within'])} |" for t in pairs["threshold"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4.5: tau/lambda/cap sweep of the clustering on one split (dev only, by design), distance matrices computed once per doc; writes the purity/coverage/contamination grid plus same- vs different-entity pairwise distance statistics for pinning the defaults.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir holding candidates (required).")
    parser.add_argument("--candidates-subdir", type=str, default="candidates", help="Candidates subdir under the run dir (default: candidates).")
    parser.add_argument("--refs-dir", type=str, default="data/derived/earnings21-conec/refs", help="Reference transcripts (default: data/derived/earnings21-conec/refs).")
    parser.add_argument("--ref-entities", type=str, default="data/derived/earnings21-conec/ref_entities", help="Reference entity index dir (default: data/derived/earnings21-conec/ref_entities).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest for split membership (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="dev", help="Split to sweep on (default: dev).")
    parser.add_argument("--taus", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55", help="Comma-separated merge thresholds (default: 0.05 to 0.55 by 0.05).")
    parser.add_argument("--lambdas", type=str, default="0.25,0.5,0.75", help="Comma-separated string-term weights (default: 0.25,0.5,0.75).")
    parser.add_argument("--dphon-caps", type=str, default="0.15,0.25,0.45", help="Comma-separated phonetic caps (default: 0.15,0.25,0.45).")
    parser.add_argument("--min-occ", type=int, default=2, help="Drop clusters with fewer mentions (default: 2).")
    parser.add_argument("--zipf-max", type=float, default=3.5, help="Common-word threshold for the merge guard (default: 3.5).")
    parser.add_argument("--out", type=str, default="", help="Output stem; empty means <run-dir>/metrics/tau_sweep_<split> (default: derived).")
    args = parser.parse_args()

    taus, lambdas, caps = floats(args.taus), floats(args.lambdas), floats(args.dphon_caps)
    run = Path(args.run_dir)
    docs = sorted(m["doc_id"] for m in read_jsonl(args.manifest) if args.split in (m["split"], "all"))
    if not docs:
        raise ValueError(f"no docs in split {args.split!r} of {args.manifest}")

    cache_dir = run / "metrics" / f"_dist_cache_{args.candidates_subdir}_{args.split}"
    os.makedirs(cache_dir, exist_ok=True)
    prepared = []
    for doc in tqdm(docs, desc="distances"):
        cands = read_jsonl(run / args.candidates_subdir / f"{doc}.jsonl")
        entities = json.loads((Path(args.ref_entities) / f"{doc}.json").read_text(encoding="utf-8"))["entities"]
        alignment = align_doc((Path(args.refs_dir) / f"{doc}.txt").read_text(encoding="utf-8"), read_hyp_text(run, "pass1", doc))
        variants = sorted({c["norm"] for c in cands})
        by_word, named = ref_word_entities(alignable_entities(alignment, entities)[0])
        # the matrices are the expensive part (g2p + panphon over every variant pair); cache them per
        # doc so an interrupted sweep or a second grid resumes in seconds
        cache_path = cache_dir / f"{doc}.npz"
        if cache_path.exists() and list(np.load(cache_path)["variants"]) == variants:
            cached = np.load(cache_path)
            mats = (cached["lev"], cached["phon"], cached["both_common"])
        else:
            mats = distance_matrices(variants, args.zipf_max)
            np.savez(cache_path, variants=np.array(variants), lev=mats[0], phon=mats[1], both_common=mats[2])
        prepared.append({"doc": doc, "cands": cands, "entities": entities, "alignment": alignment, "variants": variants,
                         "mats": mats, "word_ref": hyp_word_to_ref(alignment), "by_word": by_word, "named": named})

    rows = []
    for lam, cap, tau in tqdm(list(product(lambdas, caps, taus)), desc="grid"):
        per_doc, cluster_rows = [], []
        for d in prepared:
            clusters = build_clusters(d["doc"], d["cands"], lam, tau, args.min_occ, cap, args.zipf_max, d["mats"])
            m, r = doc_cluster_metrics(d["doc"], d["cands"], clusters, d["alignment"], d["entities"])
            per_doc.append({"doc_id": d["doc"], **m})
            cluster_rows.extend(r)
        s = summarize(per_doc, cluster_rows)
        s.pop("totals")
        rows.append({"lambda": lam, "dphon_cap": cap, "tau": tau, **s})
    pairs = pair_report(labeled_pairs(prepared), lambdas, taus, caps)

    out = Path(args.out) if args.out else run / "metrics" / f"tau_sweep_{args.split}"
    os.makedirs(out.parent, exist_ok=True)
    sha, dirty = git_sha()
    out.with_suffix(".json").write_text(json.dumps({"meta": {"split": args.split, "n_docs": len(docs), "git_sha": sha, "dirty": dirty, "argv": vars(args)},
                                                    "grid": rows, "pairs": pairs}, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(render(rows, pairs), encoding="utf-8")
    append_config(run, f"tau_sweep_{args.split}", {"argv": vars(args), "out": str(out)})
    print(render(rows, pairs))
