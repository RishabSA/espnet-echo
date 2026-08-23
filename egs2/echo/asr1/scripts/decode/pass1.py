import argparse
import json
import sys
import time
import zlib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.common.audio import load_audio, slice_audio
from scripts.common.io import append_config, create_run_dir, read_jsonl, write_jsonl
from scripts.common.whisper_engine import WhisperEngine


def decode_batch(engine: WhisperEngine, slices: list[torch.Tensor], args) -> list[list[dict]]:
    try:
        return engine.n_best_decode_batch(
            slices, num_beams=args.num_beams, num_return_sequences=args.num_return_sequences
        )
    except (torch.OutOfMemoryError, RuntimeError) as err:
        if "out of memory" not in str(err).lower():
            raise
        # one halved retry, then raise: no silent shrink loop (spec 07 section 6.3)
        print(f"warning: OOM on batch of {len(slices)}, retrying in halves", file=sys.stderr)
        half = max(1, len(slices) // 2)
        return [
            hyp
            for part in (slices[:half], slices[half:])
            if part
            for hyp in engine.n_best_decode_batch(
                part, num_beams=args.num_beams, num_return_sequences=args.num_return_sequences
            )
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S1: pass-1 chunked decode with n-best and token logprobs (spec 07 section 6.3). With --force, docs whose output already exists are skipped (resume); delete a doc's jsonl to redecode it.")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21/manifest.jsonl", help="Corpus manifest (default: data/derived/earnings21/manifest.jsonl).")
    parser.add_argument("--vad-dir", type=str, default="data/derived/earnings21/vad", help="Per-doc VAD json dir (default: data/derived/earnings21/vad).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory to write pass1/ into (required).")
    parser.add_argument("--model", type=str, default="large-v3", help="openai-whisper model name (default: large-v3).")
    parser.add_argument("--num-beams", type=int, default=8, help="Beam width (default: 8).")
    parser.add_argument("--num-return-sequences", type=int, default=8, help="Hypotheses retained per chunk (default: 8).")
    parser.add_argument("--batch-size", type=int, default=8, help="Chunks decoded per forward (default: 8).")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"], help="Which documents to decode (default: test).")
    parser.add_argument("--doc-ids", type=str, default=None, help="Comma-separated doc ids to restrict to, for smokes and replay checks (default: None).")
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty run dir, skipping completed docs (default: False).")
    args = parser.parse_args()

    run_dir = create_run_dir(args.run_dir, force=args.force)
    (run_dir / "pass1").mkdir(exist_ok=True)

    manifest = read_jsonl(args.manifest)
    if args.split != "all":
        manifest = [m for m in manifest if m["split"] == args.split]
    if args.doc_ids:
        wanted = set(args.doc_ids.split(","))
        missing = wanted - {m["doc_id"] for m in manifest}
        if missing:
            raise ValueError(f"doc ids not in manifest/split: {sorted(missing)}")
        manifest = [m for m in manifest if m["doc_id"] in wanted]
    manifest.sort(key=lambda m: m["doc_id"])

    engine = WhisperEngine(args.model)
    append_config(
        run_dir, "pass1",
        {
            # the whisper package pins checkpoints by sha, so package version + model
            # name identify the weights; n_vocab lets downstream stages rebuild the
            # exact tokenizer without loading the model
            "argv": sys.argv[1:], "model_id": args.model,
            "whisper_package": version("openai-whisper"), "n_vocab": engine.dims.n_vocab,
            "num_beams": args.num_beams, "num_return_sequences": args.num_return_sequences,
            "batch_size": args.batch_size, "split": args.split,
            "device": str(engine.device),
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )

    started = time.monotonic()
    audio_s = 0.0
    for doc in tqdm(manifest, desc="pass1"):
        out_path = run_dir / "pass1" / f"{doc['doc_id']}.jsonl"
        if out_path.exists():
            continue

        vad = json.loads((Path(args.vad_dir) / f"{doc['doc_id']}.json").read_text(encoding="utf-8"))
        overlap = vad["params"]["chunk_overlap_s"]
        audio = load_audio(doc["audio_path"])

        records = []
        chunks = vad["chunks"]
        for lo in range(0, len(chunks), args.batch_size):
            batch = chunks[lo : lo + args.batch_size]
            # overlap is decode-time context only; stored bounds stay the raw chunk
            slices = [slice_audio(audio, c["start"], c["end"], pad_s=overlap) for c in batch]
            for chunk, hyps in zip(batch, decode_batch(engine, slices, args), strict=True):
                record = {
                    "doc_id": doc["doc_id"], "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"], "end": chunk["end"], "hyps": hyps,
                }
                text = hyps[0]["text"].encode("utf-8")
                record["compression_ratio"] = len(text) / len(zlib.compress(text)) if text else 0.0
                if not hyps[0]["text"]:
                    record["empty"] = True
                records.append(record)
            audio_s += sum(c["end"] - c["start"] for c in batch)

        if [r["chunk_id"] for r in records] != list(range(len(records))):
            raise ValueError(f"{doc['doc_id']}: chunk ids not dense after decode")
        write_jsonl(out_path, records)

    wall = time.monotonic() - started
    rtf = wall / audio_s if audio_s else 0.0
    print(f"decoded {audio_s / 3600:.2f} h in {wall / 3600:.2f} h wall (RTF {rtf:.3f})")
    append_config(
        run_dir, "pass1_timing",
        {"wall_s": round(wall, 1), "decoded_audio_s": round(audio_s, 1), "rtf": round(rtf, 4)},
    )
