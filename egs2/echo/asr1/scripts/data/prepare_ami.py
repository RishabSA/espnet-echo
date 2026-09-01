import argparse
import json
import os
import shutil
from pathlib import Path
from xml.etree import ElementTree

import soundfile as sf
from tqdm import tqdm

from scripts.common.io import write_jsonl
from scripts.data.prepare_earnings import convert_audio

# the standard AMI full-corpus ASR partition (kaldi egs/ami/s5); we use only its
# dev and eval sets (~20 h): evaluation-only corpus, no training here
dev_meetings = ["ES2011a", "ES2011b", "ES2011c", "ES2011d", "IB4001", "IB4002", "IB4003", "IB4004",
                "IB4010", "IB4011", "IS1008a", "IS1008b", "IS1008c", "IS1008d", "TS3004a", "TS3004b", "TS3004c", "TS3004d"]
eval_meetings = ["EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b", "ES2004c", "ES2004d",
                 "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a", "TS3003b", "TS3003c", "TS3003d"]


def parse_words_xml(path: str | Path, speaker: str) -> tuple[list[dict], int]:
    words, despaced = [], 0
    for w in ElementTree.parse(path).getroot().iter("w"):
        if w.get("punc") == "true":
            continue
        token = (w.text or "").strip()
        start, end = w.get("starttime"), w.get("endtime")
        if not token or start is None or end is None:
            continue
        if " " in token:
            token = token.replace(" ", "")
            despaced += 1
        words.append({"token": token, "start": float(start), "end": float(end), "speaker": speaker})
    return words, despaced


def merge_speakers(per_speaker: list[list[dict]]) -> list[dict]:
    # one document token stream, interleaved by time; overlapping speech interleaves
    # word-by-word, which is the corpus's reality, not an artifact
    return sorted((w for words in per_speaker for w in words), key=lambda w: (w["start"], w["end"], w["speaker"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0: AMI IHM (Mix-Headset) preparation from the manual word annotations; entity index comes separately from build_ref_entity_index.py (spec 07 section 6.1, docs/03).")
    parser.add_argument("--annotations-dir", type=str, default="data/raw/ami/annotations", help="Extracted ami_public_manual zip with words/ (default: data/raw/ami/annotations).")
    parser.add_argument("--audio-dir", type=str, default="data/raw/ami/amicorpus", help="AMI mirror layout <meeting>/audio/<meeting>.Mix-Headset.wav (default: data/raw/ami/amicorpus).")
    parser.add_argument("--out-dir", type=str, default="data/derived/ami", help="Output dir for derived artifacts (default: data/derived/ami).")
    parser.add_argument("--meetings", type=str, default="", help="Comma-separated meeting subset, for smokes and partial downloads; empty means the full standard dev+eval lists (default: all).")
    parser.add_argument("--force", action="store_true", help="Re-encode audio and rewrite refs even if outputs exist (default: False).")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it before corpus prep")
    ann = Path(args.annotations_dir)
    out = Path(args.out_dir)
    for d in ["audio", "refs", "ref_words"]:
        os.makedirs(out / d, exist_ok=True)

    splits = {**{m: "dev" for m in dev_meetings}, **{m: "test" for m in eval_meetings}}
    if args.meetings:
        wanted = args.meetings.split(",")
        unknown = sorted(set(wanted) - set(splits))
        if unknown:
            raise ValueError(f"meetings not in the standard dev/eval lists: {unknown}")
        splits = {m: splits[m] for m in wanted}
    (out / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest, despaced_total = [], 0
    for meeting in tqdm(sorted(splits), desc="preparing meetings"):
        word_files = sorted((ann / "words").glob(f"{meeting}.*.words.xml"))
        if not word_files:
            raise FileNotFoundError(f"no word annotations for {meeting} under {ann / 'words'}")
        per_speaker = []
        for f in word_files:
            speaker = f.name.split(".")[1]
            words, despaced = parse_words_xml(f, speaker)
            per_speaker.append(words)
            despaced_total += despaced
        words = merge_speakers(per_speaker)

        (out / "refs" / f"{meeting}.txt").write_text(" ".join(w["token"] for w in words) + "\n", encoding="utf-8")
        (out / "ref_words" / f"{meeting}.json").write_text(
            json.dumps([{"start": w["start"], "end": w["end"], "speaker": w["speaker"]} for w in words]) + "\n", encoding="utf-8")

        audio_src = Path(args.audio_dir) / meeting / "audio" / f"{meeting}.Mix-Headset.wav"
        if not audio_src.exists():
            raise FileNotFoundError(f"audio missing for {meeting}: {audio_src} (download the Mix-Headset wav from the AMI mirror first)")
        wav_path = out / "audio" / f"{meeting}.wav"
        convert_audio(audio_src, wav_path, args.force)

        manifest.append({
            "doc_id": meeting, "audio_path": str(wav_path), "duration_s": sf.info(str(wav_path)).duration,
            "ref_path": str(out / "refs" / f"{meeting}.txt"), "split": splits[meeting],
            "meta": {"n_speaker_channels": len(word_files), "n_words": len(words)},
        })

    write_jsonl(out / "manifest.jsonl", manifest)
    n_dev = sum(1 for m in manifest if m["split"] == "dev")
    print(f"{len(manifest)} meetings ({n_dev} dev / {len(manifest) - n_dev} test), "
          f"{sum(m['duration_s'] for m in manifest) / 3600:.2f} h of wav, {despaced_total} tokens despaced")
    print("next: build_ref_entity_index.py --manifest data/derived/ami/manifest.jsonl --out-dir data/derived/ami/ref_entities --times-dir data/derived/ami/ref_words")
