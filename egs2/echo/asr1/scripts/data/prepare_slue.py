import argparse
import ast
import csv
import json
import os
import shutil
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from scripts.common.io import write_jsonl
from scripts.common.ref_index import entities_payload, mentions_to_entities
from scripts.data.prepare_earnings import convert_audio

# SLUE's fine-tune set is our tuning pool and its labeled dev set our test set (the SLUE
# test set is blind); documents are single utterances, per docs/03: entity-F1 calibration,
# not the consistency story
split_map = {"fine-tune": "dev", "dev": "test"}


def parse_ner(label_str: str) -> list[tuple[str, int, int]]:
    # normalized_ner holds (label, char_start, char_length) triples into normalized_text
    if not label_str or label_str in ("None", "[]"):
        return []
    return [(label, int(start), int(start) + int(length)) for label, start, length in ast.literal_eval(label_str)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0: SLUE-VoxPopuli NER subset preparation from the SLUE toolkit distribution; utterance-level documents with a one-chunk vad manifest, so segment_vad.py is not needed (docs/03).")
    parser.add_argument("--raw-dir", type=str, default="data/raw/slue/slue-voxpopuli", help="Extracted slue-voxpopuli zip with slue-voxpopuli_<split>.tsv and <split>/<id>.ogg (default: data/raw/slue/slue-voxpopuli).")
    parser.add_argument("--out-dir", type=str, default="data/derived/slue-voxpopuli", help="Output dir for derived artifacts (default: data/derived/slue-voxpopuli).")
    parser.add_argument("--force", action="store_true", help="Re-encode audio and rewrite outputs even if they exist (default: False).")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it before corpus prep")
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    for d in ["audio", "refs", "ref_entities", "vad"]:
        os.makedirs(out / d, exist_ok=True)

    manifest, n_entities = [], 0
    for slue_split, our_split in split_map.items():
        tsv = raw / f"slue-voxpopuli_{slue_split}.tsv"
        with open(tsv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in tqdm(rows, desc=f"slue {slue_split}"):
            doc = row["id"]
            text = row["normalized_text"].strip()
            tokens = text.split()
            ref_path = out / "refs" / f"{doc}.txt"
            ref_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")

            entities = mentions_to_entities(tokens, parse_ner(row["normalized_ner"]))
            (out / "ref_entities" / f"{doc}.json").write_text(
                json.dumps(entities_payload(doc, "slue_ner", entities), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            n_entities += len(entities)

            wav_path = out / "audio" / f"{doc}.wav"
            convert_audio(raw / slue_split / f"{doc}.ogg", wav_path, args.force)
            duration = sf.info(str(wav_path)).duration
            (out / "vad" / f"{doc}.json").write_text(json.dumps({
                "doc_id": doc, "backend": "utterance", "params": {"chunk_overlap_s": 0.0}, "speech_s": duration,
                "chunks": [{"chunk_id": 0, "start": 0.0, "end": round(duration, 3)}],
            }) + "\n", encoding="utf-8")

            manifest.append({
                "doc_id": doc, "audio_path": str(wav_path), "duration_s": duration, "ref_path": str(ref_path),
                "split": our_split, "meta": {"slue_split": slue_split, "speaker_id": row.get("speaker_id")},
            })

    write_jsonl(out / "manifest.jsonl", manifest)
    n_dev = sum(1 for m in manifest if m["split"] == "dev")
    print(f"{len(manifest)} utterances ({n_dev} dev / {len(manifest) - n_dev} test), "
          f"{sum(m['duration_s'] for m in manifest) / 3600:.2f} h, {n_entities} entities")
