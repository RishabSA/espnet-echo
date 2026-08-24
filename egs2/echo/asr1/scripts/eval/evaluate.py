import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

from scripts.common.align import DocAlignment, align_doc, realize_entities, tokenize
from scripts.common.io import append_config, git_sha, read_hyp_text, read_jsonl, write_jsonl
from scripts.common.normalize import normalize


def phrase_key(text: str) -> str:
    return " ".join(t.key for t in tokenize(text))


def load_bias_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def span_errors(ref_key: str, hyp_key: str | None) -> int:
    if hyp_key is None:
        return len(ref_key.split())
    if hyp_key == ref_key:
        return 0
    out = jiwer.process_words(ref_key, hyp_key)
    return out.substitutions + out.deletions + out.insertions


def alignable_entities(alignment: DocAlignment, entities: list[dict]) -> tuple[list[dict], int]:
    # Earnings-21 tags a handful of entities on bare punctuation or <inaudible> (10 occurrences
    # corpus-wide); nothing in a hypothesis can realize those, so they leave the index here
    covered = {t.word_idx for t in alignment.ref}
    kept, dropped = [], 0
    for e in entities:
        occ = [o for o in e["occurrences"] if any(i in covered for i in range(o["ref_word_span"][0], o["ref_word_span"][1] + 1))]
        dropped += len(e["occurrences"]) - len(occ)
        if occ:
            kept.append({**e, "occurrences": occ})
    return kept, dropped


def doc_occurrences(doc_id: str, alignment: DocAlignment, entities: list[dict]) -> list[dict]:
    real = realize_entities(alignment, entities)
    rows = []
    for e in entities:
        n_realized = sum(real[o["occ_id"]] is not None for o in e["occurrences"])
        for o in e["occurrences"]:
            a, b = o["ref_word_span"]
            ref_key = " ".join(t.key for t in alignment.ref if a <= t.word_idx <= b)
            r = real[o["occ_id"]]
            rows.append({
                "doc_id": doc_id, "entity_id": e["entity_id"], "category": e["category"], "occ_id": o["occ_id"],
                "n_ref": len(e["occurrences"]), "n_realized": n_realized,
                "ref_surface": o["surface"], "ref_key": ref_key, "n_ref_words": len(ref_key.split()),
                "surface": r.surface if r else None, "key": r.key if r else None,
                "hyp_span": list(r.hyp_span) if r else None,
                "errors": span_errors(ref_key, r.key if r else None),
            })
    return rows


def oracle_variant(realized: list[dict]) -> str:
    # the pool is what pass 1 actually produced; ties break toward fewer entity-word errors, then lexically
    pool = sorted({r["key"] for r in realized})
    score = {v: (sum(v == r["ref_key"] for r in realized), -sum(span_errors(r["ref_key"], v) for r in realized)) for v in pool}
    return max(pool, key=lambda v: (score[v], v))


def nc_bucket(n: int) -> str:
    # buckets follow runs/_notes/nc-distribution.md
    if n >= 10:
        return "10+"
    if n >= 5:
        return "5-9"
    return str(n)


def count_subsequence(haystack: list[str], needle: list[str]) -> int:
    n, count, i = len(needle), 0, 0
    while i + n <= len(haystack):
        if haystack[i : i + n] == needle:
            count += 1
            i += n
        else:
            i += 1
    return count


def doc_metrics(rows: list[dict], alignment: DocAlignment, entities: list[dict], ref_text: str, hyp_text: str, bias: list[str] | None) -> dict:
    ref_n, hyp_n = normalize(ref_text, "openasr"), normalize(hyp_text, "openasr")
    wer = jiwer.process_words(ref_n, hyp_n)
    m = {"wer_errors": wer.substitutions + wer.deletions + wer.insertions, "wer_n_ref": wer.substitutions + wer.deletions + wer.hits}

    c = Counter()
    by_entity = defaultdict(list)
    for r in rows:
        by_entity[r["entity_id"]].append(r)
        c["entity_errors"] += r["errors"]
        c["entity_ref_words"] += r["n_ref_words"]
        c["deleted_occurrences"] += r["key"] is None
    per_category, per_nc = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        per_category[r["category"]]["errors"] += r["errors"]
        per_category[r["category"]]["ref_words"] += r["n_ref_words"]

    oracle_errors = c["entity_errors"]
    for ent_rows in by_entity.values():
        realized = [r for r in ent_rows if r["key"] is not None]
        if len(realized) < 2:
            c["excluded_entities"] += 1
            continue
        keys = [r["key"] for r in realized]
        v = oracle_variant(realized)
        ent = Counter({
            "eligible_entities": 1, "eligible_realized": len(realized),
            "consistent_strict": len({r["surface"] for r in realized}) == 1,
            "consistent_norm": len(set(keys)) == 1,
            "pairwise_agreement": sum(keys[i] == keys[j] for i in range(len(keys)) for j in range(i + 1, len(keys))) / (len(keys) * (len(keys) - 1) / 2),
            "consistent_correct": all(r["key"] == r["ref_key"] for r in realized),
            "eligible_correct": sum(r["key"] == r["ref_key"] for r in realized),
            "oracle_correct": sum(v == r["ref_key"] for r in realized),
        })
        c.update(ent)
        per_category[ent_rows[0]["category"]].update(ent)
        per_nc[nc_bucket(len(realized))].update(ent)
        oracle_errors += sum(span_errors(r["ref_key"], v) - r["errors"] for r in realized)
    c["oracle_entity_errors"] = oracle_errors

    hyp_keys = [t.key for t in alignment.hyp]
    for e in entities:
        found = count_subsequence(hyp_keys, phrase_key(e["canonical_surface"]).split())
        c["retrieved_mentions"] += min(found, len(e["occurrences"]))
        c["ref_mentions"] += len(e["occurrences"])
    m.update(c)
    m["per_category"] = {k: dict(v) for k, v in sorted(per_category.items())}
    m["per_nc"] = {k: dict(v) for k, v in sorted(per_nc.items())}

    if bias is not None:
        # Le et al. 2021: substitutions and deletions attributed by the reference word, insertions by the
        # inserted word; computed on the same openasr alignment as WER so that B-WER and U-WER partition it
        # (fold_all keeps contractions whole while the references spell them out)
        phrases = [q for q in (normalize(p, "openasr").split() for p in bias) if q]
        words = {w for q in phrases for w in q}
        ref_words, hyp_words = wer.references[0], wer.hypotheses[0]
        b = Counter({k: 0 for k in ("b_n_ref", "b_errors", "u_n_ref", "u_errors", "bias_insertions")})
        for c in wer.alignments[0]:
            if c.type == "insert":
                inserted = hyp_words[c.hyp_start_idx : c.hyp_end_idx]
                for w in inserted:
                    b["b_errors" if w in words else "u_errors"] += 1
                # a list phrase inserted whole where the reference has nothing: the parroting pathology
                b["bias_insertions"] += sum(count_subsequence(inserted, q) for q in phrases)
                continue
            for r in range(c.ref_start_idx, c.ref_end_idx):
                side = "b" if ref_words[r] in words else "u"
                b[f"{side}_n_ref"] += 1
                b[f"{side}_errors"] += c.type != "equal"
        bias_keys = {phrase_key(p) for p in bias} - {""}
        adopted = [r for r in rows if r["key"] in bias_keys]
        b["in_list"] = len(adopted)
        b["in_list_wrong"] = sum(r["key"] != r["ref_key"] for r in adopted)
        corrupted = bias_keys - {r["ref_key"] for r in rows}
        b["realized"] = sum(r["key"] is not None for r in rows)
        b["corrupted_adopted"] = sum(r["key"] in corrupted for r in rows)
        m["bias"] = dict(b)
    return m


def transitions(rows1: list[dict], rows2: list[dict]) -> list[dict]:
    r1 = {r["occ_id"]: r for r in rows1}
    pass2_keys = defaultdict(set)
    for r in rows2:
        if r["key"] is not None:
            pass2_keys[r["entity_id"]].add(r["key"])
    out = []
    for r in rows2:
        p = r1.get(r["occ_id"])
        if p is None or p["key"] is None or r["key"] is None:
            continue
        c1, c2 = p["key"] == p["ref_key"], r["key"] == r["ref_key"]
        if c1 and c2:
            kind = "held"
        elif c1:
            kind = "damage"
        elif c2:
            kind = "repair"
        else:
            kind = "lockin" if len(pass2_keys[r["entity_id"]]) == 1 else "residual"
        out.append({"doc_id": r["doc_id"], "entity_id": r["entity_id"], "category": r["category"], "occ_id": r["occ_id"],
                    "n_ref": r["n_ref"], "ref_key": r["ref_key"], "key_pass1": p["key"], "key_pass2": r["key"], "kind": kind})
    return out


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def breakdown(c: Counter) -> dict:
    return {
        "n_eligible_entities": c["eligible_entities"],
        "ecr_strict": ratio(c["consistent_strict"], c["eligible_entities"]),
        "ecr_norm": ratio(c["consistent_norm"], c["eligible_entities"]),
        "ecr_pairwise": ratio(c["pairwise_agreement"], c["eligible_entities"]),
        "ccr": ratio(c["consistent_correct"], c["eligible_entities"]),
        "occ_correct": ratio(c["eligible_correct"], c["eligible_realized"]),
        "oracle_occ_correct": ratio(c["oracle_correct"], c["eligible_realized"]),
    }


def summarize(per_doc: list[dict], trans: list[dict] | None, baseline: list[dict] | None) -> dict:
    t = Counter()
    cat, nc = defaultdict(Counter), defaultdict(Counter)
    for d in per_doc:
        t.update({k: v for k, v in d.items() if isinstance(v, (int, float)) and k != "doc_id"})
        for k, v in d["per_category"].items():
            cat[k].update(v)
        for k, v in d["per_nc"].items():
            nc[k].update(v)
        if "bias" in d:
            t.update({f"bias_{k}": v for k, v in d["bias"].items()})
    s = {
        "n_docs": len(per_doc),
        "wer": ratio(t["wer_errors"], t["wer_n_ref"]),
        "consistency": {
            "anchoring": "reference", "n_eligible_entities": t["eligible_entities"], "n_excluded_entities": t["excluded_entities"],
            "n_deleted_occurrences": t["deleted_occurrences"], "n_unalignable_ref_occurrences": t["unalignable_ref_occurrences"],
            "ecr_strict": ratio(t["consistent_strict"], t["eligible_entities"]),
            "ecr_norm": ratio(t["consistent_norm"], t["eligible_entities"]),
            "ecr_pairwise": ratio(t["pairwise_agreement"], t["eligible_entities"]),
            "ccr": ratio(t["consistent_correct"], t["eligible_entities"]),
            "per_category": {k: breakdown(v) for k, v in sorted(cat.items()) if v["eligible_entities"]},
            "per_nc": {k: breakdown(v) for k, v in sorted(nc.items(), key=lambda kv: int(kv[0].rstrip("+").split("-")[0]))},
        },
        "oracle": {
            "n_realized_occurrences_eligible": t["eligible_realized"],
            "pass_entity_occ_correct": ratio(t["eligible_correct"], t["eligible_realized"]),
            "oracle_entity_occ_correct": ratio(t["oracle_correct"], t["eligible_realized"]),
        },
        "entity_wer": {
            "n_ref_entity_words": t["entity_ref_words"],
            "overall": ratio(t["entity_errors"], t["entity_ref_words"]),
            "oracle": ratio(t["oracle_entity_errors"], t["entity_ref_words"]),
            "per_category": {k: ratio(v["errors"], v["ref_words"]) for k, v in sorted(cat.items())},
        },
        "retrieval_recall": ratio(t["retrieved_mentions"], t["ref_mentions"]),
    }
    if "bias" in per_doc[0]:
        s["bias"] = {
            "n_biased_ref_words": t["bias_b_n_ref"],
            "b_wer": ratio(t["bias_b_errors"], t["bias_b_n_ref"]),
            "u_wer": ratio(t["bias_u_errors"], t["bias_u_n_ref"]),
            "baer": ratio(t["bias_in_list_wrong"], t["bias_in_list"]),
            "n_in_list": t["bias_in_list"],
            "bias_insertions": t["bias_bias_insertions"],
        }
        if baseline is not None:
            base = Counter()
            for d in baseline:
                base.update(d["bias"])
            # no corrupted entry adopted anywhere in the baseline means there is nothing to amplify
            biased, ref = ratio(t["bias_corrupted_adopted"], t["bias_realized"]), ratio(base["corrupted_adopted"], base["realized"])
            s["bias"]["amplification"] = 1.0 if not base["corrupted_adopted"] and not t["bias_corrupted_adopted"] else ratio(biased, ref)
    if trans is not None:
        kinds = Counter(x["kind"] for x in trans)
        n = len(trans)
        s["transitions"] = {"n_paired_occurrences": n, **{k: ratio(kinds[k], n) for k in ("held", "repair", "damage", "lockin", "residual")}}
        s["transitions"]["repair_minus_damage"] = ratio(kinds["repair"] - kinds["damage"], n)
    return s


def write_metrics(out: Path, result: dict, meta: dict) -> None:
    # summary.json is the paper's source of truth (spec 07 section 5.10); stats.py merges ci and deltas_vs into it
    os.makedirs(out, exist_ok=True)
    write_jsonl(out / "occurrences.jsonl", result["occurrences"])
    write_jsonl(out / "per_doc.jsonl", result["per_doc"])
    if result["transitions"] is not None:
        write_jsonl(out / "transitions.jsonl", result["transitions"])
    metrics = {k: v for k, v in result["summary"].items() if k != "n_docs"}
    (out / "summary.json").write_text(json.dumps({**meta, "metrics": metrics}, indent=2) + "\n", encoding="utf-8")


def evaluate(run_dir: Path, phase: str, refs_dir: Path, ref_entities: Path, doc_ids: list[str], bias: list[str] | None,
             compare: tuple[Path, str] | None, baseline: tuple[Path, str] | None) -> dict:
    occurrences, per_doc, trans, base_docs = [], [], [] if compare else None, [] if baseline else None
    for doc_id in doc_ids:
        ref_text = (refs_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        entities = json.loads((ref_entities / f"{doc_id}.json").read_text(encoding="utf-8"))["entities"]
        hyp_text = read_hyp_text(run_dir, phase, doc_id)
        alignment = align_doc(ref_text, hyp_text)
        entities, n_dropped = alignable_entities(alignment, entities)
        rows = doc_occurrences(doc_id, alignment, entities)
        occurrences.extend(rows)
        per_doc.append({"doc_id": doc_id, "unalignable_ref_occurrences": n_dropped, **doc_metrics(rows, alignment, entities, ref_text, hyp_text, bias)})
        if compare:
            prev = doc_occurrences(doc_id, align_doc(ref_text, read_hyp_text(compare[0], compare[1], doc_id)), entities)
            trans.extend(transitions(prev, rows))
        if baseline:
            base_hyp = read_hyp_text(baseline[0], baseline[1], doc_id)
            base_al = align_doc(ref_text, base_hyp)
            base_docs.append(doc_metrics(doc_occurrences(doc_id, base_al, entities), base_al, entities, ref_text, base_hyp, bias))
    return {"occurrences": occurrences, "per_doc": per_doc, "transitions": trans, "summary": summarize(per_doc, trans, base_docs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S5: reference-anchored entity evaluation of a run (WER, ECR/CCR, oracle gap, entity WER, retrieval recall, B-WER/U-WER/BAER, transitions) into <run-dir>/metrics (spec 07 sections 6.10, 7).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir to evaluate (required).")
    parser.add_argument("--phase", type=str, default="pass1", help="Subdir holding the hypotheses, pass1 or pass2 (default: pass1).")
    parser.add_argument("--refs-dir", type=str, default="data/derived/earnings21-conec/refs", help="Reference transcripts <doc>.txt; ConEC-corrected is primary per docs/03 (default: data/derived/earnings21-conec/refs).")
    parser.add_argument("--ref-entities", type=str, default="data/derived/earnings21-conec/ref_entities", help="Reference entity index <doc>.json (default: data/derived/earnings21-conec/ref_entities).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest used to select documents by split (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"], help="Which documents to evaluate (default: test).")
    parser.add_argument("--compare-run", type=str, default=None, help="Earlier run whose <phase> output pairs with this one for the transition matrix (default: None).")
    parser.add_argument("--compare-phase", type=str, default="pass1", help="Phase subdir of --compare-run (default: pass1).")
    parser.add_argument("--bias-list", type=str, default=None, help="Biasing list, one phrase per line, enabling B-WER/U-WER/BAER (default: None).")
    parser.add_argument("--baseline-run", type=str, default=None, help="No-list run for the amplification factor; needs --bias-list (default: None).")
    parser.add_argument("--baseline-phase", type=str, default="pass1", help="Phase subdir of --baseline-run (default: pass1).")
    parser.add_argument("--out", type=str, default=None, help="Output dir for metrics files (default: <run-dir>/metrics/<phase>_<split>).")
    args = parser.parse_args()

    if args.baseline_run and not args.bias_list:
        raise ValueError("--baseline-run only makes sense with --bias-list")
    manifest = read_jsonl(args.manifest)
    doc_ids = sorted(m["doc_id"] for m in manifest if args.split == "all" or m["split"] == args.split)
    run_dir = Path(args.run_dir)
    result = evaluate(
        run_dir, args.phase, Path(args.refs_dir), Path(args.ref_entities), doc_ids,
        load_bias_list(args.bias_list) if args.bias_list else None,
        (Path(args.compare_run), args.compare_phase) if args.compare_run else None,
        (Path(args.baseline_run), args.baseline_phase) if args.baseline_run else None,
    )

    out = Path(args.out) if args.out else run_dir / "metrics" / f"{args.phase}_{args.split}"
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    sha, dirty = git_sha()
    meta = {
        "run_name": run_dir.name, "phase": args.phase, "split": args.split, "corpus": Path(args.manifest).parent.name,
        "model": config.get("pass1", {}).get("model_id"), "n_docs": len(doc_ids), "docs": doc_ids,
        "git_sha": sha, "dirty": dirty, "argv": vars(args),
    }
    write_metrics(out, result, meta)
    append_config(run_dir, f"evaluate_{args.phase}_{args.split}", {"argv": vars(args), "out": str(out)})

    s = result["summary"]
    print(f"docs={s['n_docs']} wer={s['wer']:.4f} ecr_norm={s['consistency']['ecr_norm']:.4f} ccr={s['consistency']['ccr']:.4f} "
          f"oracle_occ={s['oracle']['pass_entity_occ_correct']:.4f}->{s['oracle']['oracle_entity_occ_correct']:.4f} "
          f"entity_wer={s['entity_wer']['overall']:.4f} (oracle {s['entity_wer']['oracle']:.4f})")
