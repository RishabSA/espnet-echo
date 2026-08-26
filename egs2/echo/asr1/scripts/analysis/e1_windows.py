import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts.common.align import tokenize
from scripts.common.io import append_config, create_run_dir, read_jsonl, write_jsonl

named_categories = ["PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "NORP", "EVENT", "WORK_OF_ART", "LAW"]


def stitched_words(run_dir: Path, doc_id: str) -> list[dict]:
    # the k-th whitespace token of pass1/<doc>.txt is the k-th stitched word, in chunk order
    return [w | {"chunk_id": rec["chunk_id"]} for rec in read_jsonl(run_dir / "pass1" / f"{doc_id}.words.jsonl") for w in rec["words"] if w.get("stitched")]


def doc_targets(doc_id: str, rows: list[dict], hyp_text: str, words: list[dict], chunks: dict, ref_entities: list[dict]) -> list[dict]:
    hyp_tokens = tokenize(hyp_text)
    keys_by_entity = defaultdict(set)
    for r in rows:
        if r["key"] is not None:
            keys_by_entity[r["entity_id"]].add(r["key"])
    occ_meta = {o["occ_id"]: (o["speaker"], o["ref_word_span"]) for e in ref_entities for o in e["occurrences"]}

    targets = []
    for r in rows:
        if r["category"] not in named_categories or r["n_realized"] < 2 or r["key"] is None:
            continue
        lo, hi = r["hyp_span"]
        word_idx = sorted({hyp_tokens[i].word_idx for i in range(lo, hi + 1)})
        span_words = [words[i] for i in word_idx]
        t_start, t_end = min(w["start"] for w in span_words), max(w["end"] for w in span_words)
        chunk = chunks[span_words[0]["chunk_id"]]
        speaker, ref_word_span = occ_meta[r["occ_id"]]
        # occ ids repeat across documents, so every table is keyed by the document-scoped uid
        targets.append({
            "uid": f"{doc_id}/{r['occ_id']}", "occ_id": r["occ_id"], "doc_id": doc_id, "entity_id": r["entity_id"], "category": r["category"],
            "ref_key": r["ref_key"], "ref_word_span": ref_word_span, "key": r["key"], "correct": r["key"] == r["ref_key"],
            "n_realized": r["n_realized"], "entity_consistent": len(keys_by_entity[r["entity_id"]]) == 1,
            "speaker": speaker, "word_lo": word_idx[0], "word_hi": word_idx[-1],
            "t_start": round(t_start, 3), "t_end": round(t_end, 3), "chunk_id": chunk["chunk_id"],
            "straddles_chunks": len({w["chunk_id"] for w in span_words}) > 1,
            "position": round(min(1.0, max(0.0, ((t_start + t_end) / 2 - chunk["start"]) / (chunk["end"] - chunk["start"]))), 4),
        })
    return targets


def sample_targets(targets: list[dict], n: int, seed: int) -> set[str]:
    # half from inconsistent entities, half from consistent ones; a short stratum gives its remainder to the other
    rng = random.Random(seed)
    strata = [sorted(t["uid"] for t in targets if not t["entity_consistent"]), sorted(t["uid"] for t in targets if t["entity_consistent"])]
    for s in strata:
        rng.shuffle(s)
    take = [min(n // 2, len(strata[0])), min(n - n // 2, len(strata[1]))]
    take[1] = min(len(strata[1]), n - take[0])
    take[0] = min(len(strata[0]), n - take[1])
    return set(strata[0][: take[0]]) | set(strata[1][: take[1]])


def shift_windows(chunks: list[dict], targets: list[dict], delta: float, duration: float) -> list[dict]:
    windows = []
    for c in chunks:
        s, e = c["start"] + delta, min(c["end"] + delta, duration)
        inside = [t["uid"] for t in targets if s <= t["t_start"] and t["t_end"] <= e]
        if e > s and inside:
            windows.append({"chunk_id": len(windows), "start": round(s, 3), "end": round(e, 3), "source_chunk_id": c["chunk_id"], "occ_ids": inside})
    return windows


def context_windows(targets: list[dict], context_s: float, duration: float, max_window: float, words: list[dict], prompt_words: int) -> list[dict]:
    windows = []
    for t in targets:
        # the encoder trims anything past 30 s, so the context shrinks symmetrically to fit
        half = max(0.0, min(context_s, (max_window - (t["t_end"] - t["t_start"])) / 2))
        s, e = max(0.0, t["t_start"] - half), min(duration, t["t_end"] + half)
        w = {"chunk_id": len(windows), "start": round(s, 3), "end": round(e, 3), "occ_ids": [t["uid"]]}
        if prompt_words:
            prev = [x["word"] for x in words if x["end"] <= s][-prompt_words:]
            if prev:
                w["prompt"] = " ".join(prev)
        windows.append(w)
    return windows


def write_manifest(out: Path, condition: str, doc_id: str, windows: list[dict], overlap: float) -> None:
    os.makedirs(out / "manifests" / condition, exist_ok=True)
    payload = {"doc_id": doc_id, "backend": "e1", "condition": condition, "params": {"chunk_overlap_s": overlap}, "chunks": windows}
    (out / "manifests" / condition / f"{doc_id}.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1 (M3b.2/M3b.3): build the target occurrence table and the per-condition chunk manifests (boundary shifts, context windows with and without prompt) that pass1.py --vad-dir replays on the GPU.")
    parser.add_argument("--run-dir", type=str, default="runs/e21_wlv3_pass1_espnet", help="Pass-1 run the targets come from (default: runs/e21_wlv3_pass1_espnet).")
    parser.add_argument("--metrics-dir", type=str, default=None, help="evaluate.py metrics dir with occurrences.jsonl (default: <run-dir>/metrics/pass1_<split>).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest for durations and split (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--ref-entities", type=str, default="data/derived/earnings21-conec/ref_entities", help="Reference entity index for speakers and spans (default: data/derived/earnings21-conec/ref_entities).")
    parser.add_argument("--vad-dir", type=str, default="data/derived/earnings21/vad", help="Original VAD chunk manifests (default: data/derived/earnings21/vad).")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"], help="Documents to cover (default: test).")
    parser.add_argument("--out", type=str, default="runs/e1", help="Output dir for targets.jsonl, coverage.json and manifests/ (default: runs/e1).")
    parser.add_argument("--shifts", type=str, default="2,5,10", help="Boundary offsets in seconds, comma-separated (default: 2,5,10).")
    parser.add_argument("--contexts", type=str, default="5,15", help="Context window half-widths in seconds, comma-separated; span-only is always built (default: 5,15).")
    parser.add_argument("--span-margin", type=float, default=0.2, help="Margin in seconds around the span for the span-only condition (default: 0.2).")
    parser.add_argument("--max-window", type=float, default=29.5, help="Longest window in seconds; the encoder trims past 30 s (default: 29.5).")
    parser.add_argument("--sample", type=int, default=1500, help="Occurrences sampled for the context conditions, half from inconsistent entities (default: 1500).")
    parser.add_argument("--prompt-words", type=int, default=100, help="Preceding pass-1 words supplied as prompt in the *_prompt conditions (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty output dir (default: False).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else run_dir / "metrics" / f"pass1_{args.split}"
    out = create_run_dir(args.out, force=args.force)
    manifest = [m for m in read_jsonl(args.manifest) if args.split == "all" or m["split"] == args.split]
    rows_by_doc = defaultdict(list)
    for r in read_jsonl(metrics_dir / "occurrences.jsonl"):
        rows_by_doc[r["doc_id"]].append(r)

    targets, coverage = [], defaultdict(Counter)
    shifts = [float(x) for x in args.shifts.split(",")]
    contexts = [float(x) for x in args.contexts.split(",")]
    per_doc = {}
    for m in sorted(manifest, key=lambda d: d["doc_id"]):
        doc_id = m["doc_id"]
        vad = json.loads((Path(args.vad_dir) / f"{doc_id}.json").read_text(encoding="utf-8"))
        chunks = {c["chunk_id"]: c for c in vad["chunks"]}
        words = stitched_words(run_dir, doc_id)
        ref_entities = json.loads((Path(args.ref_entities) / f"{doc_id}.json").read_text(encoding="utf-8"))["entities"]
        hyp_text = (run_dir / "pass1" / f"{doc_id}.txt").read_text(encoding="utf-8")
        doc_t = doc_targets(doc_id, rows_by_doc[doc_id], hyp_text, words, chunks, ref_entities)
        targets.extend(doc_t)
        per_doc[doc_id] = (vad, words, doc_t, m["duration_s"])

    sampled = sample_targets(targets, args.sample, args.seed)
    for t in targets:
        t["sampled_ctx"] = t["uid"] in sampled

    for doc_id, (vad, words, doc_t, duration) in per_doc.items():
        overlap = vad["params"]["chunk_overlap_s"]
        for delta in [0.0] + shifts:
            name = "base" if delta == 0 else f"shift{delta:g}"
            windows = shift_windows(vad["chunks"], doc_t, delta, duration)
            write_manifest(out, name, doc_id, windows, overlap)
            covered = {o for w in windows for o in w["occ_ids"]}
            coverage[name].update({"windows": len(windows), "audio_s": sum(w["end"] - w["start"] for w in windows), "covered": len(covered), "targets": len(doc_t)})
        ctx_targets = [t for t in doc_t if t["sampled_ctx"]]
        for label, half in [("span", args.span_margin)] + [(f"{c:g}", c) for c in contexts]:
            for prompt in (0, args.prompt_words):
                name = f"ctx_{label}" + ("_prompt" if prompt else "")
                windows = context_windows(ctx_targets, half, duration, args.max_window, words, prompt)
                write_manifest(out, name, doc_id, windows, 0.0)
                coverage[name].update({"windows": len(windows), "audio_s": sum(w["end"] - w["start"] for w in windows), "covered": len(windows), "targets": len(ctx_targets)})

    write_jsonl(out / "targets.jsonl", targets)
    (out / "coverage.json").write_text(json.dumps({k: dict(v) for k, v in coverage.items()}, indent=2) + "\n", encoding="utf-8")
    append_config(out, "e1_windows", {"argv": vars(args), "n_targets": len(targets), "n_sampled": len(sampled), "run_dir": str(run_dir)})
    print(f"targets={len(targets)} (straddling chunks: {sum(t['straddles_chunks'] for t in targets)}), sampled for context={len(sampled)}")
    for name, c in coverage.items():
        print(f"{name:16s} windows={c['windows']:5d} audio={c['audio_s'] / 3600:5.1f} h covered={c['covered']}/{c['targets']}")
