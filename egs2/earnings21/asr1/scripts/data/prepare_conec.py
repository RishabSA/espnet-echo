import argparse
import json
import os
from pathlib import Path

import jiwer
from tqdm import tqdm

from scripts.common.io import read_jsonl, write_jsonl
from scripts.common.nlp_refs import extract_entities, parse_nlp
from scripts.common.normalize import normalize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0b: ConEC-corrected references, occurrence times, and real external contexts (spec 07 section 6.1).")
    parser.add_argument("--conec-dir", type=str, default="data/raw/conec/earnings21", help="ConEC corpus dir (default: data/raw/conec/earnings21).")
    parser.add_argument("--base-derived", type=str, default="data/derived/earnings21", help="Derived dir of the uncorrected corpus, for manifest and diff (default: data/derived/earnings21).")
    parser.add_argument("--out-dir", type=str, default="data/derived/earnings21-conec", help="Output dir (default: data/derived/earnings21-conec).")
    args = parser.parse_args()

    conec = Path(args.conec_dir)
    out = Path(args.out_dir)
    # the timestamps variant carries word times and its token stream is identical
    # to nlp_references on all 44 docs (verified 2026-08-15)
    nlp_dir = conec / "transcripts" / "timestamps"
    tag_dir = conec / "transcripts" / "wer_tags"
    ctx_dir = conec / "contexts"
    for d in ["refs", "ref_entities", "conec_context"]:
        os.makedirs(out / d, exist_ok=True)

    base_manifest = {m["doc_id"]: m for m in read_jsonl(Path(args.base_derived) / "manifest.jsonl")}
    docs = sorted(p.stem for p in nlp_dir.glob("*.nlp"))
    if set(docs) != set(base_manifest):
        raise ValueError(
            f"doc mismatch: conec-only {sorted(set(docs) - set(base_manifest))}, "
            f"base-only {sorted(set(base_manifest) - set(docs))}"
        )

    manifest = []
    diff_rows = []
    for doc in tqdm(docs, desc="conec"):
        rows = parse_nlp(nlp_dir / f"{doc}.nlp")
        tag_types = json.loads((tag_dir / f"{doc}.wer_tag.json").read_text(encoding="utf-8"))
        entities, _ = extract_entities(rows, tag_types, doc)

        ref_path = out / "refs" / f"{doc}.txt"
        conec_text = " ".join(r["token"] for r in rows)
        ref_path.write_text(conec_text + "\n", encoding="utf-8")

        (out / "ref_entities" / f"{doc}.json").write_text(
            json.dumps(
                {
                    "doc_id": doc,
                    "source": "nlp_tags_conec",
                    "entities": [
                        {
                            "entity_id": e["entity_id"],
                            "category": e["category"],
                            "canonical_surface": e["canonical_surface"],
                            "occurrences": [
                                {
                                    "occ_id": o["occ_id"],
                                    "ref_word_span": o["span"],
                                    "surface": o["surface"],
                                    "speaker": o["speaker"],
                                    **({"start": o["start"], "end": o["end"]} if "start" in o else {}),
                                }
                                for o in e["occurrences"]
                            ],
                        }
                        for e in entities
                    ],
                },
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        ctx_file = ctx_dir / f"{doc}.txt"
        context_words = []
        if ctx_file.exists():
            for line in ctx_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    word, count = line.rsplit(maxsplit=1)
                    context_words.append({"word": word, "count": int(count)})
        else:
            print(f"warning: no context word list for {doc}")

        names_file = ctx_dir / "participant_names" / f"{doc}.names.txt"
        participants = []
        if names_file.exists():
            for line in names_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    name, _, role = line.partition(" - ")
                    participants.append({"name": name.strip(), "role": role.strip()})
        else:
            print(f"warning: no participant names for {doc}")

        (out / "conec_context" / f"{doc}.json").write_text(
            json.dumps(
                {"doc_id": doc, "source": "conec", "context_words": context_words,
                 "participants": participants},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        base_text = Path(base_manifest[doc]["ref_path"]).read_text(encoding="utf-8").strip()
        raw_measure = jiwer.process_words(base_text, conec_text)
        norm_measure = jiwer.process_words(
            normalize(base_text, "openasr"), normalize(conec_text, "openasr")
        )
        raw_edits = raw_measure.substitutions + raw_measure.deletions + raw_measure.insertions
        norm_edits = norm_measure.substitutions + norm_measure.deletions + norm_measure.insertions
        diff_rows.append((doc, raw_edits, norm_edits, norm_measure.wer))

        entry = dict(base_manifest[doc])
        entry["ref_path"] = str(ref_path)
        manifest.append(entry)

    write_jsonl(out / "manifest.jsonl", manifest)

    lines = [
        "# ConEC correction diff log",
        "",
        "Raw edits compare surface token streams; normalized edits and WER use the openasr policy, so they count corrections that would change metric outcomes.",
        "",
        "| doc | raw edits | normalized edits | ref-vs-ref WER |",
        "|---|---|---|---|",
    ]
    for doc, raw_edits, norm_edits, wer in diff_rows:
        lines.append(f"| {doc} | {raw_edits} | {norm_edits} | {wer:.4f} |")
    total_raw = sum(r[1] for r in diff_rows)
    total_norm = sum(r[2] for r in diff_rows)
    lines += ["", f"Totals: {total_raw} raw edits, {total_norm} normalized edits across {len(diff_rows)} docs."]
    (out / "diff-log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(manifest)} docs; {total_raw} raw edits, {total_norm} normalized edits")
    if total_norm == 0:
        raise ValueError("ConEC references are byte-identical to the raw references after normalization; corrections were not applied")
