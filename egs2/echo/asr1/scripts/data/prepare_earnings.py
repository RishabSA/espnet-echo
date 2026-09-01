import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from scripts.common.io import write_jsonl
from scripts.common.nlp_refs import extract_entities, extract_entities_from_tags, parse_nlp


def convert_audio(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        return
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
         "-loglevel", "error", str(dst)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}: {result.stderr.strip()}")


def draw_splits(meta: dict, dev_frac: float, seed: int) -> dict:
    rng = random.Random(seed)
    by_sector = {}
    for doc, m in meta.items():
        by_sector.setdefault(m["sector"], []).append(doc)

    splits = {}
    give_extra = True
    for sector in sorted(by_sector):
        docs = sorted(by_sector[sector])
        rng.shuffle(docs)
        n_dev = int(len(docs) * dev_frac)
        # sectors that do not divide evenly alternate who gets the extra document,
        # so the corpus-level split stays balanced
        if len(docs) * dev_frac != n_dev:
            if give_extra:
                n_dev += 1
            give_extra = not give_extra
        for doc in docs[:n_dev]:
            splits[doc] = "dev"
        for doc in docs[n_dev:]:
            splits[doc] = "test"
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0: Earnings-21/22 corpus preparation (spec 07 section 6.1).")
    parser.add_argument("--raw-dir", type=str, default="data/raw/speech-datasets/earnings21", help="Raw corpus dir with media/ and transcripts/ (default: data/raw/speech-datasets/earnings21).")
    parser.add_argument("--out-dir", type=str, default="data/derived/earnings21", help="Output dir for derived artifacts (default: data/derived/earnings21).")
    parser.add_argument("--corpus", type=str, default="earnings21", choices=["earnings21", "earnings22"], help="Corpus name; selects the metadata layout and entity-tag convention (default: earnings21).")
    parser.add_argument("--allow-unmatched", action="store_true", help="Skip .nlp files without a metadata row instead of raising; Earnings-22 media names are inconsistent (default: False).")
    parser.add_argument("--dev-frac", type=float, default=0.5, help="Fraction of documents in the dev split (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Split RNG seed (default: 42).")
    parser.add_argument("--force", action="store_true", help="Redraw splits and re-encode audio even if outputs exist (default: False).")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it (brew install ffmpeg) before corpus prep")

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    nlp_dir = raw / "transcripts" / "nlp_references"
    tag_dir = raw / "transcripts" / "wer_tags"
    media_dir = raw / "media"
    # E22's public release carries no named-entity tags (verified 2026-08-31, docs/06 entry 24):
    # its native normalization tags go to ref_entities_nlp/ and the canonical ref_entities/
    # comes from build_ref_entity_index.py (spaCy projection), as for AMI and TED-LIUM
    index_dirname = "ref_entities" if args.corpus == "earnings21" else "ref_entities_nlp"
    for d in ["audio", "refs", index_dirname] + (["ref_words"] if args.corpus != "earnings21" else []):
        os.makedirs(out / d, exist_ok=True)

    # both corpora stratify splits on the meta "sector" key: sector for E21, language region for E22
    meta = {}
    if args.corpus == "earnings21":
        meta_csv = raw / f"{args.corpus}-file-metadata.csv"
        with open(meta_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta[row["file_id"]] = {
                    "company": row["company_name"],
                    "sector": row["sector"],
                    "quarter": row["financial_quarter"],
                    "n_speakers": int(row["unique_speakers"]),
                    "csv_duration_s": float(row["audio_length"]),
                }
    else:
        meta_csv = raw / "metadata.csv"
        with open(meta_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                meta[row["File ID"]] = {
                    "ticker": row["Ticker Symbol"],
                    "country": row["Country by Ticker"],
                    "dialect": row["Major Dialect Family"],
                    "sector": row["Language Family + Area Based"],
                    "csv_duration_s": float(row["File Length (seconds)"]),
                }

    docs = sorted(p.stem for p in nlp_dir.glob("*.nlp"))
    if not docs:
        raise FileNotFoundError(f"no .nlp files under {nlp_dir}")
    missing_meta = [d for d in docs if d not in meta]
    if missing_meta and args.allow_unmatched:
        print(f"skipping {len(missing_meta)} .nlp files without metadata rows: {missing_meta}")
        docs = [d for d in docs if d in meta]
    elif missing_meta:
        raise ValueError(f"docs without metadata rows: {missing_meta} (pass --allow-unmatched to skip)")
    missing_files = sorted(set(meta) - set(docs))
    if missing_files:
        raise ValueError(f"metadata rows without .nlp files: {missing_files}")

    splits_path = out / "splits.json"
    if splits_path.exists() and not args.force:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        print(f"reusing existing {splits_path}")
    else:
        splits = draw_splits(meta, args.dev_frac, args.seed)
        splits_path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_dev = sum(1 for s in splits.values() if s == "dev")
    print(f"splits: {n_dev} dev / {len(splits) - n_dev} test")

    manifest = []
    skipped_tags_total = 0
    for doc in tqdm(docs, desc="preparing docs"):
        audio_src = media_dir / f"{doc}.mp3"
        if not audio_src.exists():
            raise FileNotFoundError(f"audio missing for {doc}: {audio_src}")
        wav_path = out / "audio" / f"{doc}.wav"
        convert_audio(audio_src, wav_path, args.force)
        duration_s = sf.info(str(wav_path)).duration

        rows = parse_nlp(nlp_dir / f"{doc}.nlp")
        if args.corpus == "earnings21":
            tag_types = json.loads((tag_dir / f"{doc}.wer_tag.json").read_text(encoding="utf-8"))
            entities, stats = extract_entities(rows, tag_types, doc)
        else:
            entities, stats = extract_entities_from_tags(rows, doc)
        skipped_tags_total += stats["n_skipped_tag_ids"]

        # raw tokens, space-joined: token index == whitespace index, the invariant
        # ref_entities spans rely on; metric-time normalization handles punctuation
        ref_path = out / "refs" / f"{doc}.txt"
        ref_path.write_text(" ".join(r["token"] for r in rows) + "\n", encoding="utf-8")
        if args.corpus != "earnings21":
            (out / "ref_words" / f"{doc}.json").write_text(
                json.dumps([{"start": r["ts"], "end": r["end_ts"], "speaker": r["speaker"]} for r in rows]) + "\n", encoding="utf-8")

        (out / index_dirname / f"{doc}.json").write_text(
            json.dumps(
                {
                    "doc_id": doc,
                    "source": "nlp_tags",
                    "entities": [
                        {
                            "entity_id": e["entity_id"],
                            "category": e["category"],
                            "canonical_surface": e["canonical_surface"],
                            "occurrences": [
                                {"occ_id": o["occ_id"], "ref_word_span": o["span"],
                                 "surface": o["surface"], "speaker": o["speaker"]}
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

        manifest.append(
            {
                "doc_id": doc,
                "audio_path": str(wav_path),
                "duration_s": duration_s,
                "ref_path": str(ref_path),
                "split": splits[doc],
                "meta": {k: v for k, v in meta[doc].items() if k != "csv_duration_s"},
            }
        )

    write_jsonl(out / "manifest.jsonl", manifest)

    wav_h = sum(m["duration_s"] for m in manifest) / 3600
    csv_h = sum(meta[d]["csv_duration_s"] for d in docs) / 3600
    print(f"{len(manifest)} docs, {wav_h:.2f} h of wav (metadata says {csv_h:.2f} h)")
    print(f"skipped wer_tag ids: {skipped_tags_total}")
    if abs(wav_h - csv_h) / csv_h > 0.01:
        raise ValueError(f"wav duration {wav_h:.2f} h deviates from metadata {csv_h:.2f} h by more than 1%")
