import argparse
import json
import os
from pathlib import Path

import spacy
from tqdm import tqdm

from scripts.analysis.e1_windows import stitched_words
from scripts.common.align import align_doc
from scripts.common.io import append_config, git_sha, read_hyp_text, read_jsonl
from scripts.mine.cluster_variants import build_clusters
from scripts.mine.eval_clusters import doc_cluster_metrics, summarize
from scripts.mine.mine_candidates import all_signals, load_stoplist, mine_doc, ner_mentions

# Analysis 5: each signal alone, the union, and the union without the boilerplate stoplist
signal_sets = {**{s: {s} for s in all_signals}, "union": set(all_signals), "union_nostop": set(all_signals)}
table_cols = ["candidates_per_doc", "candidate_norms_per_doc", "candidate_recall_occ", "candidate_recall_entity", "clusters_per_doc", "named_cluster_share",
              "coverage_occ", "purity", "contamination", "vote_damage_rate", "vote_repair_rate", "oracle_damage_rate", "oracle_repair_rate", "list_precision", "list_recall", "list_recall_any_variant"]


def fmt(v: float | None) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4.6 / Analysis 5: mining ablation over signal sets at pinned clustering parameters, spaCy run once per doc; writes <run-dir>/metrics/mining_ablation_<split>.{json,md}.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir with pass1 transcripts and word timings (required).")
    parser.add_argument("--refs-dir", type=str, default="data/derived/earnings21-conec/refs", help="Reference transcripts (default: data/derived/earnings21-conec/refs).")
    parser.add_argument("--ref-entities", type=str, default="data/derived/earnings21-conec/ref_entities", help="Reference entity index dir (default: data/derived/earnings21-conec/ref_entities).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest for split membership (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="dev", help="Split to ablate on (default: dev).")
    parser.add_argument("--model", type=str, default="en_core_web_lg", help="spaCy model (default: en_core_web_lg).")
    parser.add_argument("--zipf-max", type=float, default=3.5, help="rarity threshold and common-word guard (default: 3.5).")
    parser.add_argument("--conf-max", type=float, default=-0.80, help="lowconf threshold (default: -0.80).")
    parser.add_argument("--max-span-words", type=int, default=4, help="Longest rarity/caps run emitted as one candidate (default: 4).")
    parser.add_argument("--stoplist", type=str, default="scripts/mine/stoplist_earnings.txt", help="Boilerplate stoplist (default: scripts/mine/stoplist_earnings.txt).")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.25, help="Pinned string-term weight (default: 0.25).")
    parser.add_argument("--tau", type=float, default=0.10, help="Pinned merge threshold (default: 0.10).")
    parser.add_argument("--min-occ", type=int, default=2, help="Pinned minimum cluster size (default: 2).")
    parser.add_argument("--dphon-cap", type=float, default=0.25, help="Pinned phonetic cap (default: 0.25).")
    args = parser.parse_args()

    docs = sorted(m["doc_id"] for m in read_jsonl(args.manifest) if args.split in (m["split"], "all"))
    if not docs:
        raise ValueError(f"no docs in split {args.split!r} of {args.manifest}")
    nlp = spacy.load(args.model, disable=["lemmatizer"])
    stoplist = load_stoplist(args.stoplist)
    run = Path(args.run_dir)

    prepared = []
    for doc in tqdm(docs, desc="tagging"):
        hyp_text = read_hyp_text(run, "pass1", doc)
        words = stitched_words(run, doc)
        if len(hyp_text.split()) != len(words):
            raise ValueError(f"{doc}: {len(hyp_text.split())} transcript tokens vs {len(words)} stitched words")
        prepared.append({"doc": doc, "words": words, "mentions": ner_mentions(nlp, hyp_text),
                         "entities": json.loads((Path(args.ref_entities) / f"{doc}.json").read_text(encoding="utf-8"))["entities"],
                         "alignment": align_doc((Path(args.refs_dir) / f"{doc}.txt").read_text(encoding="utf-8"), hyp_text)})

    rows = []
    for name, signals in tqdm(signal_sets.items(), desc="signal sets"):
        per_doc, cluster_rows = [], []
        for d in prepared:
            cands = mine_doc(d["doc"], d["words"], d["mentions"], signals, args.zipf_max, args.conf_max, args.max_span_words,
                             set() if name == "union_nostop" else stoplist)
            clusters = build_clusters(d["doc"], cands, args.lam, args.tau, args.min_occ, args.dphon_cap, args.zipf_max)
            m, r = doc_cluster_metrics(d["doc"], cands, clusters, d["alignment"], d["entities"])
            per_doc.append({"doc_id": d["doc"], **m})
            cluster_rows.extend(r)
        s = summarize(per_doc, cluster_rows)
        s.pop("totals")
        rows.append({"signals": name, **s})

    lines = ["| signals | " + " | ".join(table_cols) + " |", "|" + "---|" * (len(table_cols) + 1)]
    lines += ["| " + r["signals"] + " | " + " | ".join(fmt(r[c]) for c in table_cols) + " |" for r in rows]
    table = "\n".join(lines) + "\n"
    out = run / "metrics" / f"mining_ablation_{args.split}"
    os.makedirs(out.parent, exist_ok=True)
    sha, dirty = git_sha()
    out.with_suffix(".json").write_text(json.dumps({"meta": {"split": args.split, "n_docs": len(docs), "git_sha": sha, "dirty": dirty, "argv": vars(args)}, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(table, encoding="utf-8")
    append_config(run, f"mining_ablation_{args.split}", {"argv": vars(args), "out": str(out)})
    print(table)
