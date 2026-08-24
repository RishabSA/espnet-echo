import argparse
import json
import os
from pathlib import Path

named_categories = ["PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "NORP", "EVENT", "WORK_OF_ART", "LAW"]
transition_kinds = ["held", "repair", "damage", "lockin", "residual", "repair_minus_damage"]


def load(paths: list[str | Path]) -> list[dict]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def fmt(x: float | None, ci: list[float] | None = None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    s = f"{x:.{digits}f}"
    if ci:
        s += f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"
    return s


def label(s: dict) -> str:
    return f"{s['run_name']} ({s['phase']}, {s['split']})"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def headline_table(summaries: list[dict], floor: dict | None) -> str:
    floor_ccr = floor["metrics"]["consistency"]["ccr"] if floor else None
    rows = []
    for s in summaries + ([floor] if floor else []):
        m, ci, c = s["metrics"], s.get("ci", {}), s["metrics"]["consistency"]
        rows.append([
            label(s) + (" (noise floor)" if s is floor else ""), str(s["n_docs"]), str(c["n_eligible_entities"]),
            fmt(m["wer"], ci.get("wer")), fmt(c["ecr_strict"], ci.get("ecr_strict")), fmt(c["ecr_norm"], ci.get("ecr_norm")),
            fmt(c["ecr_pairwise"], ci.get("ecr_pairwise")), fmt(c["ccr"], ci.get("ccr")), fmt(c["ccr"] / floor_ccr) if floor_ccr else "n/a",
            fmt(m["oracle"]["pass_entity_occ_correct"], ci.get("occ_correct")), fmt(m["oracle"]["oracle_entity_occ_correct"], ci.get("oracle_occ_correct")),
            fmt(m["entity_wer"]["overall"], ci.get("entity_wer")), fmt(m["entity_wer"]["oracle"], ci.get("oracle_entity_wer")),
            fmt(m["retrieval_recall"], ci.get("retrieval_recall")),
        ])
    return table(["run", "docs", "eligible", "WER", "ECR strict", "ECR norm", "ECR pair", "CCR", "CCR / floor", "occ correct", "oracle occ", "entity WER", "oracle entity WER", "retrieval"], rows)


def category_table(summaries: list[dict], metric: str) -> str:
    def cell(s: dict, cat: str) -> str:
        if metric == "entity_wer":
            return fmt(s["metrics"]["entity_wer"]["per_category"].get(cat))
        v = s["metrics"]["consistency"]["per_category"].get(cat)
        return fmt(v[metric]) if v else "n/a"

    cats = [c for c in named_categories if any(c in s["metrics"]["entity_wer"]["per_category"] for s in summaries)]
    return table(["category"] + [label(s) for s in summaries], [[c] + [cell(s, c) for s in summaries] for c in cats])


def nc_table(summaries: list[dict], metric: str) -> str:
    buckets = []
    for s in summaries:
        buckets += [b for b in s["metrics"]["consistency"]["per_nc"] if b not in buckets]
    rows = [[b] + [fmt(s["metrics"]["consistency"]["per_nc"].get(b, {}).get(metric)) for s in summaries] for b in buckets]
    return table(["N_c"] + [label(s) for s in summaries], rows)


def transitions_table(summaries: list[dict]) -> str:
    rows = [[label(s), str(s["metrics"]["transitions"]["n_paired_occurrences"])] + [fmt(s["metrics"]["transitions"][k]) for k in transition_kinds] for s in summaries]
    return table(["run", "paired occurrences"] + transition_kinds, rows)


def bias_table(summaries: list[dict]) -> str:
    rows = []
    for s in summaries:
        b, ci = s["metrics"]["bias"], s.get("ci", {})
        rows.append([label(s), str(b["n_biased_ref_words"]), fmt(b["b_wer"], ci.get("b_wer")), fmt(b["u_wer"], ci.get("u_wer")), fmt(b["baer"], ci.get("baer")),
                     str(b["n_in_list"]), str(b["bias_insertions"]), fmt(b.get("amplification"))])
    return table(["run", "biased ref words", "B-WER", "U-WER", "BAER", "adopted", "list insertions", "A"], rows)


def deltas_table(summaries: list[dict]) -> str:
    rows = []
    for s in summaries:
        d = s["deltas_vs"]
        for k in ("wer", "ecr_norm", "ccr", "entity_wer", "occ_correct"):
            p = d[k]["wilcoxon_p"]
            rows.append([label(s), d["run"], k, fmt(d[k]["point"], d[k]["ci95"], 4), "n/a" if p is None else f"{p:.3f}"])
    return table(["run", "vs", "metric", "difference [95% CI]", "Wilcoxon p"], rows)


def render(summaries: list[dict], floor: dict | None) -> str:
    parts = ["## Headline (reference-anchored; CIs are document-level bootstrap)", headline_table(summaries, floor)]
    for metric, title in (("ccr", "CCR"), ("ecr_norm", "ECR norm"), ("entity_wer", "entity WER")):
        parts += [f"## {title} by named-entity category", category_table(summaries, metric)]
    parts += ["## CCR by N_c", nc_table(summaries, "ccr")]
    if with_trans := [s for s in summaries if s["metrics"].get("transitions")]:
        parts += ["## Transitions (pass 1 to pass 2)", transitions_table(with_trans)]
    if with_bias := [s for s in summaries if s["metrics"].get("bias")]:
        parts += ["## Biasing-list metrics", bias_table(with_bias)]
    if with_deltas := [s for s in summaries if s.get("deltas_vs")]:
        parts += ["## Paired differences", deltas_table(with_deltas)]
    sources = "\n".join(f"- {label(s)}: git {s['git_sha'][:10]}{' (dirty)' if s.get('dirty') else ''}" for s in summaries + ([floor] if floor else []))
    parts += ["## Sources", sources]
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate markdown result tables from summary.json files; no number reaches the paper except through here (spec 07 sections 5.10, 11).")
    parser.add_argument("--summaries", type=str, nargs="+", required=True, help="summary.json files, one per run to tabulate (required).")
    parser.add_argument("--floor", type=str, default=None, help="summary.json of the reference noise-floor evaluation; adds its row and CCR relative to it (default: None).")
    parser.add_argument("--out", type=str, required=True, help="Markdown file to write (required).")
    args = parser.parse_args()

    text = render(load(args.summaries), load([args.floor])[0] if args.floor else None)
    os.makedirs(Path(args.out).parent, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
