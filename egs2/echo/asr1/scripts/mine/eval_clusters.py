import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path

from scripts.analysis.nc_distribution import named_categories
from scripts.common.align import DocAlignment, align_doc, realize
from scripts.common.io import append_config, git_sha, read_hyp_text, read_jsonl, write_jsonl
from scripts.eval.evaluate import alignable_entities, phrase_key
from scripts.mine.mine_candidates import word_norm


def hyp_word_to_ref(alignment: DocAlignment) -> dict[int, list[int]]:
    # hypothesis word index -> reference token indices it aligns to (substitutions and equals only)
    out = {}
    for i, h in enumerate(alignment.ref_to_hyp):
        if h is not None:
            out.setdefault(alignment.hyp[h].word_idx, []).append(i)
    return out


def ref_word_entities(entities: list[dict]) -> tuple[dict[int, str], set[int]]:
    # reference word index -> entity id (first tag wins where the index tags a span twice, e.g.
    # YEAR and CARDINAL on the same year), plus the set of words under a named-category entity
    by_word, named = {}, set()
    for e in entities:
        for o in e["occurrences"]:
            for i in range(o["ref_word_span"][0], o["ref_word_span"][1] + 1):
                by_word.setdefault(i, e["entity_id"])
                if e["category"] in named_categories:
                    named.add(i)
    return by_word, named


def mention_labels(mention: dict, alignment: DocAlignment, word_ref: dict[int, list[int]], by_word: dict[int, str], named: set[int]) -> tuple[str | None, str | None, bool]:
    # (reference spelling, reference entity id, touches a named entity). The spelling is the
    # possessive-stripped fold_all form of the aligned reference tokens joined without spaces,
    # so "Kowalski's" and "Kowalski" label alike and "Kai-Fu" matches "Kaifu": this is what a
    # cluster must not mix, because canonicalizing a mixed cluster overwrites one true spelling
    # with another. Entity ids are the index's coarser view (spec 07 section 5.3 grouping), kept
    # as a secondary label. A pure insertion aligns to nothing and gets no labels.
    a, b = mention["word_span"]
    ref_idx = sorted({i for w in range(a, b + 1) for i in word_ref.get(w, ())})
    if not ref_idx:
        return None, None, False
    spelling = "".join(word_norm(alignment.ref[i].raw) for i in ref_idx) or None
    words = [alignment.ref[i].word_idx for i in ref_idx]
    ids = Counter(by_word[w] for w in words if w in by_word)
    entity = ids.most_common(1)[0][0] if ids else None
    return spelling, entity, any(w in named for w in words)


def span_words(alignment: DocAlignment, hyp_span: tuple[int, int]) -> set[int]:
    return {alignment.hyp[i].word_idx for i in range(hyp_span[0], hyp_span[1] + 1)}


def doc_cluster_metrics(doc_id: str, cands: list[dict], clusters: dict, alignment: DocAlignment, entities: list[dict]) -> tuple[dict, list[dict]]:
    entities, _ = alignable_entities(alignment, entities)
    word_ref = hyp_word_to_ref(alignment)
    by_word, named = ref_word_entities(entities)
    cand_by_id = {c["occ_id"]: c for c in cands}
    category = {e["entity_id"]: e["category"] for e in entities}

    c = Counter()
    rows, covered, mined, any_variant, named_sizes = [], set(), set(), set(), []
    for cl in clusters["clusters"]:
        mentions = [cand_by_id[o] for o in cl["occ_ids"]]
        labels = [mention_labels(m, alignment, word_ref, by_word, named) for m in mentions]
        spellings, ids = Counter(l[0] for l in labels), Counter(l[1] for l in labels)
        n = len(mentions)
        real_spellings = sorted(k for k in spellings if k is not None)
        real_ids = sorted(k for k in ids if k is not None)
        contaminated, entity_contaminated = len(real_spellings) >= 2, len(real_ids) >= 2
        is_named = any(l[2] for l in labels)
        scope = ["all", "named"] if is_named else ["all"]
        # what a plurality-vote canonicalization (A0 over canonicalizer (a)) would do to this
        # cluster: a mention is correct when its own spelling matches the reference spelling it
        # aligns to; damage overwrites a correct mention, repair fixes a wrong one
        plurality_norm = Counter(m["norm"] for m in mentions).most_common(1)[0][0].replace(" ", "")
        labeled = [(m["norm"].replace(" ", ""), spelling) for m, (spelling, _, _) in zip(mentions, labels, strict=True) if spelling is not None and is_named]
        # the pool-restricted oracle picks the cluster variant that leaves most mentions correct:
        # the ceiling any arbiter can reach on these clusters, and the damage a contaminated
        # cluster forces on even a perfect one
        pool = sorted({norm for norm, _ in labeled})
        oracle_norm = max(pool, key=lambda v: (sum(v == spelling for _, spelling in labeled), v)) if pool else None
        for norm, spelling in labeled:
            correct_now = norm == spelling
            c["vote_labeled"] += 1
            c["vote_correct_now"] += correct_now
            for tag, choice in (("vote", plurality_norm), ("oracle", oracle_norm)):
                correct_after = choice == spelling
                c[f"{tag}_damage"] += correct_now and not correct_after
                c[f"{tag}_repair"] += correct_after and not correct_now
        for sc in scope:
            c[f"cluster_mentions_{sc}"] += n
            c[f"pure_mentions_{sc}"] += spellings.most_common(1)[0][1]
            c[f"contaminated_mentions_{sc}"] += n * contaminated
            c[f"entity_pure_mentions_{sc}"] += ids.most_common(1)[0][1]
            c[f"entity_contaminated_mentions_{sc}"] += n * entity_contaminated
            c[f"n_clusters_{sc}"] += 1
            c[f"n_contaminated_clusters_{sc}"] += contaminated
        if is_named:
            named_sizes.append(n)
        for m in mentions:
            covered.update(range(m["word_span"][0], m["word_span"][1] + 1))
        mined.add(phrase_key(cl["plurality"]))
        any_variant.update(phrase_key(v) for v in cl["variants"])
        rows.append({"doc_id": doc_id, "cluster_id": cl["cluster_id"], "n_occ": n, "n_variants": len(cl["variants"]), "named": is_named,
                     "purity": spellings.most_common(1)[0][1] / n, "contaminated": contaminated,
                     "spellings": {str(k): v for k, v in spellings.most_common()}, "entity_ids": {str(k): v for k, v in ids.most_common()},
                     "categories": sorted({category[k] for k in real_ids})})

    cand_words = set()
    for m in cands:
        cand_words.update(range(m["word_span"][0], m["word_span"][1] + 1))

    # the method's targets are named entities with two or more reference occurrences
    oracle = set()
    for e in entities:
        if e["category"] not in named_categories:
            continue
        target = len(e["occurrences"]) >= 2
        if target:
            oracle.add(phrase_key(e["canonical_surface"]))
        hit_cand = hit_cov = False
        for o in e["occurrences"]:
            r = realize(alignment, o["ref_word_span"])
            if r is None:
                c["target_occ_deleted"] += target
                continue
            widx = span_words(alignment, r.hyp_span)
            in_cand, in_cov = bool(widx & cand_words), bool(widx & covered)
            hit_cand, hit_cov = hit_cand or in_cand, hit_cov or in_cov
            if target:
                c["target_occ"] += 1
                c["target_occ_candidate"] += in_cand
                c["target_occ_covered"] += in_cov
        if target:
            c["target_entities"] += 1
            c["target_entities_candidate"] += hit_cand
            c["target_entities_covered"] += hit_cov
    c["oracle_list"] = len(oracle)
    c["mined_list"] = len(mined)
    c["list_hits"] = len(oracle & mined)
    c["list_hits_any_variant"] = len(oracle & any_variant)
    c["n_candidates"] = len(cands)
    c["n_candidate_norms"] = len({m["norm"] for m in cands})
    return {**c, "named_sizes": named_sizes}, rows


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def summarize(per_doc: list[dict], rows: list[dict]) -> dict:
    t = Counter()
    for d in per_doc:
        t.update({k: v for k, v in d.items() if isinstance(v, int | float)})
    named_sizes = [s for d in per_doc for s in d.get("named_sizes", [])]
    all_sizes = [r["n_occ"] for r in rows]
    return {
        "purity": ratio(t["pure_mentions_named"], t["cluster_mentions_named"]),
        "contamination": ratio(t["contaminated_mentions_named"], t["cluster_mentions_named"]),
        "entity_purity": ratio(t["entity_pure_mentions_named"], t["cluster_mentions_named"]),
        "entity_contamination": ratio(t["entity_contaminated_mentions_named"], t["cluster_mentions_named"]),
        "purity_all": ratio(t["pure_mentions_all"], t["cluster_mentions_all"]),
        "contamination_all": ratio(t["contaminated_mentions_all"], t["cluster_mentions_all"]),
        "vote_damage_rate": ratio(t["vote_damage"], t["vote_labeled"]),
        "vote_repair_rate": ratio(t["vote_repair"], t["vote_labeled"]),
        "vote_correct_now": ratio(t["vote_correct_now"], t["vote_labeled"]),
        "oracle_damage_rate": ratio(t["oracle_damage"], t["vote_labeled"]),
        "oracle_repair_rate": ratio(t["oracle_repair"], t["vote_labeled"]),
        "coverage_occ": ratio(t["target_occ_covered"], t["target_occ"]),
        "coverage_entity": ratio(t["target_entities_covered"], t["target_entities"]),
        "candidate_recall_occ": ratio(t["target_occ_candidate"], t["target_occ"]),
        "candidate_recall_entity": ratio(t["target_entities_candidate"], t["target_entities"]),
        "list_precision": ratio(t["list_hits"], t["mined_list"]),
        "list_recall": ratio(t["list_hits"], t["oracle_list"]),
        "list_recall_any_variant": ratio(t["list_hits_any_variant"], t["oracle_list"]),
        "candidates_per_doc": ratio(t["n_candidates"], len(per_doc)),
        "candidate_norms_per_doc": ratio(t["n_candidate_norms"], len(per_doc)),
        "clusters_per_doc": ratio(t["n_clusters_all"], len(per_doc)),
        "named_cluster_share": ratio(t["n_clusters_named"], t["n_clusters_all"]),
        "median_n_occ": statistics.median(all_sizes) if all_sizes else None,
        "share_n_occ_2": ratio(sum(s == 2 for s in all_sizes), len(all_sizes)),
        "named_median_n_occ": statistics.median(named_sizes) if named_sizes else None,
        "named_share_n_occ_2": ratio(sum(s == 2 for s in named_sizes), len(named_sizes)),
        "named_share_n_occ_5plus": ratio(sum(s >= 5 for s in named_sizes), len(named_sizes)),
        "totals": dict(t),
    }


def evaluate(run_dir: str | Path, cand_subdir: str, clus_subdir: str, refs_dir: str | Path, ref_entities_dir: str | Path, doc_ids: list[str]) -> dict:
    run = Path(run_dir)
    per_doc, rows, params = [], [], None
    for doc in doc_ids:
        cands = read_jsonl(run / cand_subdir / f"{doc}.jsonl")
        clusters = json.loads((run / clus_subdir / f"{doc}.json").read_text(encoding="utf-8"))
        params = clusters["params"]
        ref_text = (Path(refs_dir) / f"{doc}.txt").read_text(encoding="utf-8")
        entities = json.loads((Path(ref_entities_dir) / f"{doc}.json").read_text(encoding="utf-8"))["entities"]
        alignment = align_doc(ref_text, read_hyp_text(run, "pass1", doc))
        m, r = doc_cluster_metrics(doc, cands, clusters, alignment, entities)
        per_doc.append({"doc_id": doc, **m})
        rows.extend(r)
    summary = summarize(per_doc, rows)
    for d in per_doc:
        d.pop("named_sizes")
    return {"summary": summary, "params": params, "per_doc": per_doc, "clusters": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2c: score clusters against the reference entity index (purity, coverage, contamination, candidate recall, bias-list precision/recall) into <run-dir>/metrics/<clusters-subdir>_<split> (spec 07 section 6.7).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir holding candidates and clusters (required).")
    parser.add_argument("--candidates-subdir", type=str, default="candidates", help="Candidates subdir under the run dir (default: candidates).")
    parser.add_argument("--clusters-subdir", type=str, default="clusters", help="Clusters subdir under the run dir (default: clusters).")
    parser.add_argument("--refs-dir", type=str, default="data/derived/earnings21-conec/refs", help="Reference transcripts (default: data/derived/earnings21-conec/refs).")
    parser.add_argument("--ref-entities", type=str, default="data/derived/earnings21-conec/ref_entities", help="Reference entity index dir (default: data/derived/earnings21-conec/ref_entities).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest for split membership (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="dev", help="Split to score, or all (default: dev).")
    parser.add_argument("--docs", type=str, default="", help="Comma-separated doc ids overriding the manifest lookup (default: manifest split).")
    args = parser.parse_args()

    docs = args.docs.split(",") if args.docs else sorted(m["doc_id"] for m in read_jsonl(args.manifest) if args.split in (m["split"], "all"))
    if not docs:
        raise ValueError(f"no docs in split {args.split!r} of {args.manifest}")
    result = evaluate(args.run_dir, args.candidates_subdir, args.clusters_subdir, args.refs_dir, args.ref_entities, docs)

    out = Path(args.run_dir) / "metrics" / f"{args.clusters_subdir}_{args.split}"
    os.makedirs(out, exist_ok=True)
    sha, dirty = git_sha()
    summary = {"meta": {"run_dir": args.run_dir, "split": args.split, "n_docs": len(docs), "clusters_subdir": args.clusters_subdir,
                        "git_sha": sha, "dirty": dirty, "argv": vars(args)},
               "params": result["params"], "metrics": result["summary"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out / "per_doc.jsonl", result["per_doc"])
    write_jsonl(out / "clusters.jsonl", result["clusters"])
    append_config(args.run_dir, f"eval_{args.clusters_subdir}_{args.split}", {"argv": vars(args), "out": str(out)})
    s = result["summary"]
    print(f"{len(docs)} docs, named clusters: purity {s['purity']:.3f}, contamination {s['contamination']:.3f}, vote damage {s['vote_damage_rate']:.3f} / repair {s['vote_repair_rate']:.3f}, oracle damage {s['oracle_damage_rate']:.3f} / repair {s['oracle_repair_rate']:.3f}; "
          f"coverage occ {s['coverage_occ']:.3f} / entity {s['coverage_entity']:.3f}, candidate recall occ {s['candidate_recall_occ']:.3f}, "
          f"list P {s['list_precision']:.3f} R {s['list_recall']:.3f} (any variant {s['list_recall_any_variant']:.3f}); "
          f"{s['clusters_per_doc']:.1f} clusters/doc, named share {s['named_cluster_share']:.2f}, named median N_c {s['named_median_n_occ']}, "
          f"named share N_c=2 {s['named_share_n_occ_2']:.2f}")
