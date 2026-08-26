import argparse
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from scripts.analysis.e1_windows import stitched_words
from scripts.common.align import DocAlignment, align_doc, realize
from scripts.common.io import append_config, git_sha, read_jsonl, write_jsonl
from scripts.eval.stats import bootstrap

shift_conditions = ["shift2", "shift5", "shift10"]
decoder_conditions = ["greedy", "repeat"]
context_conditions = ["ctx_span", "ctx_5", "ctx_15", "ctx_span_prompt", "ctx_5_prompt", "ctx_15_prompt"]


def build_pairs(targets: list[dict]) -> list[dict]:
    by_entity = defaultdict(list)
    for t in targets:
        by_entity[(t["doc_id"], t["entity_id"])].append(t)
    pairs = []
    for occs in by_entity.values():
        for a, b in combinations(sorted(occs, key=lambda t: t["t_start"]), 2):
            pairs.append({
                "doc_id": a["doc_id"], "entity_id": a["entity_id"], "category": a["category"], "n_realized": a["n_realized"],
                "occ_i": a["uid"], "occ_j": b["uid"], "differ": a["key"] != b["key"],
                "cross_speaker": a["speaker"] != b["speaker"], "cross_chunk": a["chunk_id"] != b["chunk_id"],
                "pos_diff": round(abs(a["position"] - b["position"]), 4), "time_gap": round(b["t_start"] - a["t_end"], 3),
                "correct_i": a["correct"], "correct_j": b["correct"],
            })
    expected = sum(len(v) * (len(v) - 1) // 2 for v in by_entity.values())
    if len(pairs) != expected:
        raise ValueError(f"pair count {len(pairs)} does not match the occurrence counts ({expected})")
    return pairs


def doc_counts(items: list[dict], docs: list[str], num: str, den: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = {d: 0.0 for d in docs}
    m = {d: 0.0 for d in docs}
    for x in items:
        if den is None or x[den]:
            m[x["doc_id"]] += 1
            n[x["doc_id"]] += bool(x[num])
    return np.array([n[d] for d in docs]), np.array([m[d] for d in docs])


def rate(items: list[dict], docs: list[str], num: str, den: str | None, n_boot: int, seed: int) -> dict:
    n, m = doc_counts(items, docs, num, den)
    return {"n": int(m.sum()), **bootstrap(n, m, n_boot, seed)}


def phi_bootstrap(pairs: list[dict], docs: list[str], n_boot: int, seed: int) -> dict:
    # correctness correlation between the two occurrences of a pair, from per-document 2x2 counts
    cells = {d: np.zeros(4) for d in docs}
    for p in pairs:
        cells[p["doc_id"]][2 * int(p["correct_i"]) + int(p["correct_j"])] += 1
    table = np.array([cells[d] for d in docs])  # shape: (n_docs, 4): [wrong-wrong, wrong-correct, correct-wrong, correct-correct]

    def phi(t: np.ndarray) -> float:
        n00, n01, n10, n11 = t
        den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        return float((n11 * n00 - n10 * n01) / den) if den else float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(docs), size=(n_boot, len(docs)))
    est = np.array([phi(table[i].sum(axis=0)) for i in idx])
    est = est[np.isfinite(est)]
    return {"point": phi(table.sum(axis=0)), "ci95": [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))] if len(est) else None}


def speaker_analysis(pairs: list[dict], docs: list[str], n_boot: int, seed: int) -> dict:
    out = {}
    for stratum, keep in (("all", lambda p: True), ("cross_chunk", lambda p: p["cross_chunk"]), ("same_chunk", lambda p: not p["cross_chunk"])):
        sub = [p for p in pairs if keep(p)]
        for spk in ("same", "cross"):
            group = [p | {"one_wrong": p["correct_i"] != p["correct_j"], "both_wrong": not p["correct_i"] and not p["correct_j"], "any_wrong": not (p["correct_i"] and p["correct_j"])}
                     for p in sub if p["cross_speaker"] == (spk == "cross")]
            out[f"{stratum}/{spk}_speaker"] = {
                "n_pairs": len(group),
                "differ": rate(group, docs, "differ", None, n_boot, seed),
                "both_wrong_given_any_wrong": rate(group, docs, "both_wrong", "any_wrong", n_boot, seed),
                "phi_correctness": phi_bootstrap(group, docs, n_boot, seed) if group else None,
            }
    return out


def fit_mixed_model(pairs: list[dict]) -> dict:
    df = pd.DataFrame(pairs)
    for col in ("differ", "cross_speaker", "cross_chunk"):
        df[col] = df[col].astype(int)
    df["ent"] = df["doc_id"] + "/" + df["entity_id"]
    model = BinomialBayesMixedGLM.from_formula("differ ~ cross_speaker + cross_chunk + pos_diff", {"doc": "0 + C(doc_id)", "ent": "0 + C(ent)"}, df)
    result = model.fit_vb()
    fixed = {name: {"mean": float(m), "sd": float(s), "odds_ratio": float(np.exp(m))} for name, m, s in zip(model.exog_names, result.fe_mean, result.fe_sd, strict=True)}
    random_sd = {name: float(np.exp(v)) for name, v in zip(model.vcp_names, result.vcp_mean, strict=True)}
    return {"n_pairs": len(df), "fixed_effects": fixed, "random_effect_sd": random_sd, "method": "BinomialBayesMixedGLM.fit_vb"}


def window_realizations(base: DocAlignment, base_words: list[dict], ref_words: list[str], window: dict, hyp_text: str, targets: list[dict]) -> dict:
    # the reference slice for a window is whatever reference the base run's words inside it aligned to
    hyp_to_ref = {h: i for i, h in enumerate(base.ref_to_hyp) if h is not None}
    inside = [i for i, t in enumerate(base.hyp) if base_words[t.word_idx]["start"] >= window["start"] and base_words[t.word_idx]["end"] <= window["end"]]
    ref_idx = [hyp_to_ref[i] for i in inside if i in hyp_to_ref]
    if not ref_idx:
        return {t["uid"]: None for t in targets}
    a, b = base.ref[min(ref_idx)].word_idx, base.ref[max(ref_idx)].word_idx
    local = align_doc(" ".join(ref_words[a : b + 1]), hyp_text)
    out = {}
    for t in targets:
        sa, sb = t["ref_word_span"]
        if sa < a or sb > b:
            out[t["uid"]] = None
            continue
        r = realize(local, [sa - a, sb - a])
        out[t["uid"]] = r.key if r else None
    return out


def condition_keys(cond_run: Path, manifest_dir: Path, doc_id: str, base: DocAlignment, base_words: list[dict], ref_words: list[str], targets_by_occ: dict, source_records: dict | None = None) -> dict:
    manifest = json.loads((manifest_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
    if source_records is not None:
        texts = {w["chunk_id"]: source_records[w["source_chunk_id"]] for w in manifest["chunks"]}
    else:
        texts = {r["chunk_id"]: r["hyps"][0]["text"] for r in read_jsonl(cond_run / "pass1" / f"{doc_id}.jsonl")}
    keys = {}
    for w in manifest["chunks"]:
        keys.update(window_realizations(base, base_words, ref_words, w, texts[w["chunk_id"]], [targets_by_occ[o] for o in w["occ_ids"]]))
    return keys


def flip_table(base_keys: dict, cond_keys: dict, targets_by_occ: dict) -> list[dict]:
    rows = []
    for occ, k in cond_keys.items():
        b = base_keys.get(occ)
        if b is None or k is None:
            continue
        ref = targets_by_occ[occ]["ref_key"]
        rows.append({"doc_id": targets_by_occ[occ]["doc_id"], "uid": occ, "flip": k != b, "repair": b != ref and k == ref, "damage": b == ref and k != ref,
                     "wrong_to_wrong": b != ref and k != ref and k != b, "base_wrong": b != ref, "cond_correct": k == ref})
    return rows


def flip_summary(rows: list[dict], docs: list[str], n_boot: int, seed: int) -> dict:
    return {"n_paired": len(rows), "flip": rate(rows, docs, "flip", None, n_boot, seed), "repair": rate(rows, docs, "repair", None, n_boot, seed),
            "damage": rate(rows, docs, "damage", None, n_boot, seed), "wrong_to_wrong": rate(rows, docs, "wrong_to_wrong", None, n_boot, seed),
            "correct": rate(rows, docs, "cond_correct", None, n_boot, seed), "repair_given_base_wrong": rate(rows, docs, "repair", "base_wrong", n_boot, seed)}


def systematic_share(targets: list[dict], wrong_everywhere: dict, docs: list[str], n_boot: int, seed: int) -> dict:
    # an entity is systematic when every realized occurrence stays wrong under every condition it was
    # observed in; the share is over entities with at least one error in the base decode
    by_entity = defaultdict(list)
    for t in targets:
        by_entity[(t["doc_id"], t["entity_id"])].append(t)
    ents = []
    for (doc_id, entity_id), occs in by_entity.items():
        any_error = any(not t["correct"] for t in occs)
        systematic = all(wrong_everywhere[t["uid"]] for t in occs)
        ents.append({"doc_id": doc_id, "entity_id": entity_id, "any_error": any_error, "systematic": any_error and systematic})
    return {"n_entities_with_error": sum(e["any_error"] for e in ents), "share": rate(ents, docs, "systematic", "any_error", n_boot, seed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1 decomposition (M3b.1, M3b.4 to M3b.6): occurrence-pair table, speaker analysis, mixed-effects fit, intervention flip rates and the systematic share; conditions whose run dirs are absent are skipped and listed.")
    parser.add_argument("--e1-dir", type=str, default="runs/e1", help="Dir written by e1_windows.py (default: runs/e1).")
    parser.add_argument("--run-dir", type=str, default="runs/e21_wlv3_pass1_espnet", help="Base pass-1 run (default: runs/e21_wlv3_pass1_espnet).")
    parser.add_argument("--refs-dir", type=str, default="data/derived/earnings21-conec/refs", help="Reference transcripts (default: data/derived/earnings21-conec/refs).")
    parser.add_argument("--conditions-root", type=str, default="runs", help="Dir holding the condition runs e1_<condition> (default: runs).")
    parser.add_argument("--model-metrics", type=str, default="", help="Comma-separated evaluate.py metrics dirs of the other model sizes, for the systematic share (default: none).")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Document-level bootstrap resamples (default: 10000).")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap and hand-check sampling seed (default: 42).")
    parser.add_argument("--handcheck", type=int, default=10, help="Pairs listed in handcheck.md for listening (default: 10).")
    args = parser.parse_args()

    e1 = Path(args.e1_dir)
    run_dir = Path(args.run_dir)
    targets = read_jsonl(e1 / "targets.jsonl")
    targets_by_occ = {t["uid"]: t for t in targets}
    docs = sorted({t["doc_id"] for t in targets})
    summary = {"n_targets": len(targets), "n_docs": len(docs), "bootstrap": args.bootstrap, "seed": args.seed}

    pairs = build_pairs(targets)
    write_jsonl(e1 / "pairs.jsonl", pairs)
    summary["pairs"] = {"n": len(pairs), "differ": rate(pairs, docs, "differ", None, args.bootstrap, args.seed),
                        "cross_speaker_share": rate(pairs, docs, "cross_speaker", None, args.bootstrap, args.seed),
                        "cross_chunk_share": rate(pairs, docs, "cross_chunk", None, args.bootstrap, args.seed)}
    summary["speaker"] = speaker_analysis(pairs, docs, args.bootstrap, args.seed)
    summary["mixed_model"] = fit_mixed_model(pairs)

    rng = random.Random(args.seed)
    lines = ["# E1 hand-check pairs", "", "Listen to both occurrences and confirm the realizations (seed 42 sample).", ""]
    for p in rng.sample(pairs, min(args.handcheck, len(pairs))):
        a, b = targets_by_occ[p["occ_i"]], targets_by_occ[p["occ_j"]]
        lines.append(f"- {p['doc_id']} {p['category']} ref `{a['ref_key']}`: {a['t_start']}-{a['t_end']} s (spk {a['speaker']}) -> `{a['key']}`; {b['t_start']}-{b['t_end']} s (spk {b['speaker']}) -> `{b['key']}`")
    (e1 / "handcheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # base realizations through the same windowed extraction every condition uses
    base_keys, cond_keys, present, absent = {}, defaultdict(dict), [], []
    per_doc = {}
    for doc_id in docs:
        ref_words = (Path(args.refs_dir) / f"{doc_id}.txt").read_text(encoding="utf-8").split()
        base_words = stitched_words(run_dir, doc_id)
        base_al = align_doc(" ".join(ref_words), (run_dir / "pass1" / f"{doc_id}.txt").read_text(encoding="utf-8"))
        records = {r["chunk_id"]: r["hyps"][0]["text"] for r in read_jsonl(run_dir / "pass1" / f"{doc_id}.jsonl")}
        per_doc[doc_id] = (ref_words, base_words, base_al, records)
        base_keys.update(condition_keys(run_dir, e1 / "manifests" / "base", doc_id, base_al, base_words, ref_words, targets_by_occ, source_records=records))
    agree = [{"doc_id": targets_by_occ[o]["doc_id"], "agree": k == targets_by_occ[o]["key"]} for o, k in base_keys.items() if k is not None]
    summary["base"] = {"n_covered": sum(k is not None for k in base_keys.values()), "windowed_vs_stitched_agreement": rate(agree, docs, "agree", None, args.bootstrap, args.seed)}

    for cond in shift_conditions + decoder_conditions + context_conditions:
        cond_run = Path(args.conditions_root) / f"e1_{cond}"
        manifest_dir = e1 / "manifests" / ("base" if cond in decoder_conditions else cond)
        if not all((cond_run / "pass1" / f"{d}.jsonl").exists() for d in docs):
            absent.append(cond)
            continue
        present.append(cond)
        for doc_id in docs:
            ref_words, base_words, base_al, _ = per_doc[doc_id]
            cond_keys[cond].update(condition_keys(cond_run, manifest_dir, doc_id, base_al, base_words, ref_words, targets_by_occ))
        summary.setdefault("conditions", {})[cond] = flip_summary(flip_table(base_keys, cond_keys[cond], targets_by_occ), docs, args.bootstrap, args.seed)
    if "repeat" in present:
        identical = []
        for doc_id in docs:
            ours = {r["chunk_id"]: r["hyps"][0]["text"] for r in read_jsonl(Path(args.conditions_root) / "e1_repeat" / "pass1" / f"{doc_id}.jsonl")}
            manifest = json.loads((e1 / "manifests" / "base" / f"{doc_id}.json").read_text(encoding="utf-8"))
            identical += [{"doc_id": doc_id, "same": ours[w["chunk_id"]] == per_doc[doc_id][3][w["source_chunk_id"]]} for w in manifest["chunks"]]
        summary["conditions"]["repeat"]["byte_identical_chunks"] = rate(identical, docs, "same", None, args.bootstrap, args.seed)
    shifts_present = [c for c in shift_conditions if c in present]
    if shifts_present:
        any_flip = []
        for occ, base_key in base_keys.items():
            ks = [k for k in (cond_keys[c].get(occ) for c in shifts_present) if k is not None]
            if base_key is not None and ks:
                any_flip.append({"doc_id": targets_by_occ[occ]["doc_id"], "flip": any(k != base_key for k in ks)})
        summary["boundary_shift_any"] = rate(any_flip, docs, "flip", None, args.bootstrap, args.seed)

    model_dirs = [Path(x) for x in args.model_metrics.split(",") if x]
    models = {}
    for d in model_dirs:
        models[d.as_posix()] = {f"{r['doc_id']}/{r['occ_id']}": r["key"] == r["ref_key"] for r in read_jsonl(d / "occurrences.jsonl") if f"{r['doc_id']}/{r['occ_id']}" in targets_by_occ}
    wrong_everywhere = {}
    for t in targets:
        occ = t["uid"]
        conds = [cond_keys[c].get(occ) for c in shifts_present]
        wrong = [not t["correct"]] + [k != t["ref_key"] for k in conds if k is not None] + [not m.get(occ, False) for m in models.values()]
        wrong_everywhere[occ] = all(wrong)
    if models:
        shared = [{"doc_id": t["doc_id"], "shared": all(not m.get(t["uid"], False) for m in models.values()), "base_wrong": not t["correct"]} for t in targets]
        summary["models"] = {"dirs": list(models), "wrong_in_all_sizes_given_base_wrong": rate(shared, docs, "shared", "base_wrong", args.bootstrap, args.seed)}
    summary["systematic"] = systematic_share(targets, wrong_everywhere, docs, args.bootstrap, args.seed)
    # occurrence level: among occurrences wrong in the base run, the share wrong under every observed condition;
    # unlike the entity-level S it is not bounded by the base run's own consistency
    occ_rows = [{"doc_id": t["doc_id"], "systematic": wrong_everywhere[t["uid"]], "base_wrong": not t["correct"]} for t in targets]
    summary["systematic"]["occurrence_share"] = rate(occ_rows, docs, "systematic", "base_wrong", args.bootstrap, args.seed)
    summary["systematic"]["conditions_used"] = ["base"] + shifts_present + list(models)
    summary["conditions_absent"] = absent

    sha, dirty = git_sha()
    summary["git_sha"], summary["dirty"] = sha, dirty
    (e1 / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    append_config(e1, "decompose", {"argv": vars(args), "present": present, "absent": absent})

    s = summary
    print(f"targets={s['n_targets']} pairs={s['pairs']['n']} differ={s['pairs']['differ']['point']:.4f}")
    for k in ("all/same_speaker", "all/cross_speaker", "cross_chunk/same_speaker", "cross_chunk/cross_speaker"):
        v = s["speaker"][k]
        print(f"{k:26s} n={v['n_pairs']:5d} differ={v['differ']['point']:.4f} [{v['differ']['ci95'][0]:.4f}, {v['differ']['ci95'][1]:.4f}] phi={v['phi_correctness']['point']:.3f}")
    for name, fe in s["mixed_model"]["fixed_effects"].items():
        print(f"mixed {name:14s} {fe['mean']:+.3f} (sd {fe['sd']:.3f}) OR={fe['odds_ratio']:.2f}")
    print("random-effect sd:", {k: round(v, 3) for k, v in s["mixed_model"]["random_effect_sd"].items()})
    print(f"base windowed vs stitched agreement: {s['base']['windowed_vs_stitched_agreement']['point']:.4f} over {s['base']['n_covered']} covered")
    for cond, c in s.get("conditions", {}).items():
        print(f"{cond:16s} n={c['n_paired']:5d} flip={c['flip']['point']:.4f} repair={c['repair']['point']:.4f} damage={c['damage']['point']:.4f} correct={c['correct']['point']:.4f}")
    if "boundary_shift_any" in s:
        print(f"boundary shift, any offset: flip={s['boundary_shift_any']['point']:.4f} [{s['boundary_shift_any']['ci95'][0]:.4f}, {s['boundary_shift_any']['ci95'][1]:.4f}]")
    print(f"systematic share S={s['systematic']['share']['point']:.4f} over {s['systematic']['n_entities_with_error']} entities with errors "
          f"(occurrence-level {s['systematic']['occurrence_share']['point']:.4f}), using {s['systematic']['conditions_used']}")
    if absent:
        print(f"conditions absent: {absent}")
