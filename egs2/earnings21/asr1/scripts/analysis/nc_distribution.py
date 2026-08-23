import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.common.nlp_refs import extract_entities, parse_nlp

# spaCy-style named-entity categories the method targets (mining list, spec 6.5);
# everything else (DATE, CARDINAL, CONTRACTION, ...) is reported but not decisive
named_categories = {"PERSON", "ORG", "GPE", "PRODUCT", "FAC", "NORP", "LOC", "EVENT", "WORK_OF_ART"}


def quartile_row(counts: list[int]) -> str:
    rep = [c for c in counts if c >= 2]
    if not rep:
        return "| 0 | - | - | - | - |"
    q1, q2, q3 = statistics.quantiles(rep, n=4) if len(rep) > 1 else (rep[0], rep[0], rep[0])
    return f"| {len(rep)} | {q1:.1f} | {q2:.1f} | {q3:.1f} | {max(rep)} |"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1.0: N_c distribution from .nlp entity annotations, no audio, no model.")
    parser.add_argument("--raw-dir", type=str, default="data/raw/speech-datasets/earnings21", help="Earnings raw dir containing transcripts/ (default: data/raw/speech-datasets/earnings21).")
    parser.add_argument("--out", type=str, default="runs/_notes/nc-distribution.md", help="Output markdown path (default: runs/_notes/nc-distribution.md).")
    args = parser.parse_args()

    nlp_dir = Path(args.raw_dir) / "transcripts" / "nlp_references"
    tag_dir = Path(args.raw_dir) / "transcripts" / "wer_tags"
    docs = sorted(p.stem for p in nlp_dir.glob("*.nlp"))
    if not docs:
        raise FileNotFoundError(f"no .nlp files under {nlp_dir}")

    per_category = defaultdict(list)  # category -> list of N_c across all docs
    per_doc_named = {}
    hist_named = Counter()
    skipped_total = 0

    for doc in docs:
        rows = parse_nlp(nlp_dir / f"{doc}.nlp")
        tag_types = json.loads((tag_dir / f"{doc}.wer_tag.json").read_text(encoding="utf-8"))
        entities, stats = extract_entities(rows, tag_types, doc)
        skipped_total += stats["n_skipped_tag_ids"]

        named = [e for e in entities if e["category"] in named_categories]
        per_doc_named[doc] = {
            "entities": len(named),
            "repeated": sum(1 for e in named if len(e["occurrences"]) >= 2),
            "occ_in_repeated": sum(len(e["occurrences"]) for e in named if len(e["occurrences"]) >= 2),
            "occ_total": sum(len(e["occurrences"]) for e in named),
        }
        for entity in entities:
            n_c = len(entity["occurrences"])
            per_category[entity["category"]].append(n_c)
            if entity["category"] in named_categories:
                bucket = str(n_c) if n_c < 5 else ("5-9" if n_c < 10 else "10+")
                hist_named[bucket] += 1

    named_counts = [n for cat in named_categories for n in per_category.get(cat, [])]
    all_counts = [n for counts in per_category.values() for n in counts]

    lines = [
        "# M1.0: N_c distribution (Earnings-21 reference entities)",
        "",
        f"Parsed {len(docs)} documents; {skipped_total} span ids skipped (missing from wer_tag dicts).",
        "Grouping rule: (entity_type, fold_all surface) within document, no alias merging (spec 07 section 5.3), so every number here is conservative.",
        "",
        "## The decision numbers (named-entity categories only)",
        "",
    ]

    n_named = len(named_counts)
    n_named_rep = sum(1 for n in named_counts if n >= 2)
    occ_named = sum(named_counts)
    occ_named_rep = sum(n for n in named_counts if n >= 2)
    rep = [n for n in named_counts if n >= 2]
    q1, q2, q3 = statistics.quantiles(rep, n=4)
    lines += [
        f"- named entities per document: median {statistics.median(d['entities'] for d in per_doc_named.values()):.0f} (min {min(d['entities'] for d in per_doc_named.values())}, max {max(d['entities'] for d in per_doc_named.values())})",
        f"- entities with N_c >= 2: {n_named_rep}/{n_named} ({n_named_rep / n_named:.1%})",
        f"- occurrences belonging to entities with N_c >= 2: {occ_named_rep}/{occ_named} ({occ_named_rep / occ_named:.1%}) <- the identifiable mass the method can act on",
        f"- N_c among repeated entities: q1 {q1:.1f}, median {q2:.1f}, q3 {q3:.1f}, max {max(rep)}",
        "",
        "N_c histogram (named categories): " + ", ".join(f"{k}: {hist_named[k]}" for k in ["1", "2", "3", "4", "5-9", "10+"]),
        "",
        "## Per category (all 27 corpus categories)",
        "",
        "| category | entities | occurrences | N_c>=2 entities | q1 | median | q3 | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cat in sorted(per_category, key=lambda c: -sum(per_category[c])):
        counts = per_category[cat]
        marker = " *" if cat in named_categories else ""
        lines.append(f"| {cat}{marker} | {len(counts)} | {sum(counts)} " + quartile_row(counts))
    lines += [
        "",
        "`*` = named-entity category targeted by the method.",
        "",
        "## All categories pooled",
        "",
        f"- entities: {len(all_counts)}, occurrences: {sum(all_counts)}, N_c >= 2 share of entities: {sum(1 for n in all_counts if n >= 2) / len(all_counts):.1%}",
        "",
        "## Per document (named categories)",
        "",
        "| doc | named entities | repeated | occ in repeated / total |",
        "|---|---|---|---|",
    ]
    for doc, d in sorted(per_doc_named.items()):
        lines.append(f"| {doc} | {d['entities']} | {d['repeated']} | {d['occ_in_repeated']}/{d['occ_total']} |")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"\nwrote {args.out}")
