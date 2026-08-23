import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

from scripts.common.audio import load_audio
from scripts.common.io import read_jsonl

sr = 16000


def fine_split(audio: torch.Tensor, model, start: float, end: float, chunk_max_s: float) -> list[dict]:
    # a single VAD region longer than the chunk budget: look for internal silences
    # with a finer silence threshold and split at the longest one, recursively;
    # hard-cut only when the region is genuinely unbroken
    if end - start <= chunk_max_s:
        return [{"start": start, "end": end, "hard_cut": False}]

    lo, hi = int(start * sr), int(end * sr)
    fine = get_speech_timestamps(
        audio[lo:hi], model, sampling_rate=sr,
        min_silence_duration_ms=100, min_speech_duration_ms=250,
    )
    gaps = []
    for a, b in itertools.pairwise(fine):
        gaps.append((a["end"], b["start"]))
    if not gaps:
        cut = start + chunk_max_s
        return [{"start": start, "end": cut, "hard_cut": True}] + fine_split(audio, model, cut, end, chunk_max_s)

    gap_start, gap_end = max(gaps, key=lambda g: g[1] - g[0])
    mid = start + (gap_start + gap_end) / 2 / sr
    return fine_split(audio, model, start, mid, chunk_max_s) + fine_split(audio, model, mid, end, chunk_max_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1: silero VAD segmentation into decode chunks (spec 07 section 6.2).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21/manifest.jsonl", help="Corpus manifest (default: data/derived/earnings21/manifest.jsonl).")
    parser.add_argument("--out-dir", type=str, default="data/derived/earnings21/vad", help="Output dir for per-doc VAD json (default: data/derived/earnings21/vad).")
    parser.add_argument("--chunk-max-s", type=float, default=28.0, help="Maximum chunk length in seconds, before overlap extension (default: 28.0).")
    parser.add_argument("--chunk-overlap-s", type=float, default=0.5, help="Overlap applied by pass 1 at slice time, recorded here for the stitcher (default: 0.5).")
    parser.add_argument("--min-silence-s", type=float, default=0.4, help="Minimum silence separating speech regions (default: 0.4).")
    parser.add_argument("--min-speech-s", type=float, default=0.5, help="Minimum speech region length (default: 0.5).")
    parser.add_argument("--force", action="store_true", help="Recompute docs whose VAD json already exists (default: False).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_silero_vad()
    manifest = read_jsonl(args.manifest)

    total_speech = 0.0
    total_audio = 0.0
    n_hard_cuts = 0
    n_chunks = 0
    for doc in tqdm(manifest, desc="vad"):
        out_path = Path(args.out_dir) / f"{doc['doc_id']}.json"
        total_audio += doc["duration_s"]
        if out_path.exists() and not args.force:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            total_speech += payload["speech_s"]
            n_hard_cuts += sum(c.get("hard_cut", False) for c in payload["chunks"])
            n_chunks += len(payload["chunks"])
            continue

        audio = load_audio(doc["audio_path"])
        regions = get_speech_timestamps(
            audio, model, sampling_rate=sr,
            min_silence_duration_ms=int(args.min_silence_s * 1000),
            min_speech_duration_ms=int(args.min_speech_s * 1000),
        )
        speech_s = sum((r["end"] - r["start"]) / sr for r in regions)

        # split oversized regions first, then greedily pack neighbours into chunks
        pieces = []
        for r in regions:
            pieces.extend(fine_split(audio, model, r["start"] / sr, r["end"] / sr, args.chunk_max_s))

        chunks = []
        for piece in pieces:
            if chunks and not chunks[-1]["hard_cut"] and piece["end"] - chunks[-1]["start"] <= args.chunk_max_s:
                chunks[-1]["end"] = piece["end"]
                chunks[-1]["hard_cut"] = piece["hard_cut"]
            else:
                chunks.append(dict(piece))

        payload = {
            "doc_id": doc["doc_id"],
            "backend": "silero",
            "params": {
                "chunk_max_s": args.chunk_max_s, "chunk_overlap_s": args.chunk_overlap_s,
                "min_silence_s": args.min_silence_s, "min_speech_s": args.min_speech_s,
            },
            "speech_s": speech_s,
            "chunks": [
                {"chunk_id": i, "start": round(c["start"], 3), "end": round(c["end"], 3),
                 **({"hard_cut": True} if c["hard_cut"] else {})}
                for i, c in enumerate(chunks)
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        total_speech += speech_s
        n_hard_cuts += sum(c["hard_cut"] for c in chunks)
        n_chunks += len(chunks)

        too_long = [c for c in chunks if c["end"] - c["start"] > args.chunk_max_s + 1e-6]
        if too_long:
            raise ValueError(f"{doc['doc_id']}: {len(too_long)} chunks exceed {args.chunk_max_s}s: {too_long[:3]}")

    print(f"{n_chunks} chunks, {n_hard_cuts} hard cuts")
    print(f"speech {total_speech / 3600:.2f} h of {total_audio / 3600:.2f} h audio ({total_speech / total_audio:.1%})")
