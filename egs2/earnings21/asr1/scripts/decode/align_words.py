import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.common.audio import load_audio
from scripts.common.io import append_config, read_jsonl, write_jsonl
from scripts.common.whisper_engine import pick_device

sr = 16000
# MMS_FA vocabulary is lowercase roman letters plus apostrophe; anything that
# normalizes to nothing (numbers, symbols) aligns as the star token
_nonroman_re = re.compile(r"[^a-z']")


def group_tokens_into_words(tokens: list[dict], text: str, whisper_tok, where: str) -> list[list[dict]]:
    groups = []
    for token in tokens:
        if token["tok"].startswith("Ġ") or not groups:
            groups.append([])
        groups[-1].append(token)
    words = [whisper_tok.decode([t["id"] for t in g]).strip() for g in groups]
    if words != text.split():
        raise ValueError(f"{where}: token grouping {words[:8]}... does not match text words {text.split()[:8]}...")
    return groups


def word_score(spans) -> float:
    total = sum(s.end - s.start for s in spans)
    return sum(s.score * (s.end - s.start) for s in spans) / total if total else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S1b: MMS_FA word alignment of pass-1 1-best plus overlap-aware stitching to pass1/<doc>.txt (spec 07 sections 6.4, 6.2).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir containing pass1/ (required).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21/manifest.jsonl", help="Corpus manifest for audio paths (default: data/derived/earnings21/manifest.jsonl).")
    parser.add_argument("--vad-dir", type=str, default="data/derived/earnings21/vad", help="VAD dir, for the decode-time overlap value (default: data/derived/earnings21/vad).")
    parser.add_argument("--force", action="store_true", help="Recompute docs whose words.jsonl already exists (default: False).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    whisper_tok = AutoTokenizer.from_pretrained(
        config["pass1"]["model_id"], revision=config["pass1"]["model_revision"]
    )

    device = pick_device()
    bundle = torchaudio.pipelines.MMS_FA
    fa_model = bundle.get_model(with_star=True).to(device)
    fa_model.eval()
    fa_tokenizer = bundle.get_tokenizer()
    fa_aligner = bundle.get_aligner()

    manifest = {m["doc_id"]: m for m in read_jsonl(args.manifest)}
    docs = sorted(p.stem for p in (run_dir / "pass1").glob("*.jsonl") if not p.stem.endswith(".words"))

    n_words_total = 0
    unalignable = []  # CTC-infeasible 1-bests (decode loops): excluded, warned about
    degenerate = []  # (doc, chunk) needing escalation; whisperx/whisper_dtw rungs are
    # not implemented yet, so any entry here fails the run at the end, loudly
    append_config(
        run_dir, "align_words",
        {"argv": sys.argv[1:], "aligner": "mms_fa", "device": str(device),
         "started_utc": datetime.now(UTC).isoformat(timespec="seconds")},
    )

    for doc_id in tqdm(docs, desc="align"):
        words_path = run_dir / "pass1" / f"{doc_id}.words.jsonl"
        text_path = run_dir / "pass1" / f"{doc_id}.txt"
        if words_path.exists() and text_path.exists() and not args.force:
            continue

        records = read_jsonl(run_dir / "pass1" / f"{doc_id}.jsonl")
        vad = json.loads((Path(args.vad_dir) / f"{doc_id}.json").read_text(encoding="utf-8"))
        overlap = vad["params"]["chunk_overlap_s"]
        audio = load_audio(manifest[doc_id]["audio_path"])

        per_chunk_words = []
        chunk_labels = {}
        for record in records:
            where = f"{doc_id}#c{record['chunk_id']}"
            chunk_labels[record["chunk_id"]] = "mms_fa"
            if record.get("empty"):
                per_chunk_words.append([])
                continue

            hyp = record["hyps"][0]
            groups = group_tokens_into_words(hyp["tokens"], hyp["text"], whisper_tok, where)
            words = hyp["text"].split()
            fa_words = [_nonroman_re.sub("", w.lower()) or "*" for w in words]

            # same slice the decoder saw, so aligned times share its clock
            lo = max(0, int((record["start"] - overlap) * sr))
            hi = min(audio.shape[0], int((record["end"] + overlap) * sr))
            with torch.inference_mode():
                emission, _ = fa_model(audio[lo:hi].unsqueeze(dim=0).to(device))
            ratio = (hi - lo) / emission.shape[1]  # samples per emission frame

            try:
                spans = fa_aligner(emission[0].cpu(), fa_tokenizer(fa_words))
            except Exception as err:  # noqa: BLE001 — classified below, never silent
                # CTC alignment is infeasible when the text has more characters than
                # the audio has emission frames: that is a pass-1 decode pathology
                # (repetition loop; see the chunk's compression_ratio), not an aligner
                # defect, so it is excluded from the transcript and reported as a
                # warning instead of triggering aligner escalation
                if sum(len(w) for w in fa_words) >= emission.shape[1]:
                    unalignable.append((where, f"cr={record['compression_ratio']:.2f}"))
                    chunk_labels[record["chunk_id"]] = "unalignable_output"
                else:
                    degenerate.append((where, f"aligner error: {err}"))
                per_chunk_words.append([])
                continue

            chunk_words = []
            for idx, (word, group, word_spans) in enumerate(zip(words, groups, spans, strict=True)):
                start = lo / sr + word_spans[0].start * ratio / sr
                end = lo / sr + word_spans[-1].end * ratio / sr
                chunk_words.append(
                    {
                        "idx": idx, "word": word,
                        "start": round(start, 3), "end": round(end, 3),
                        "logprob": sum(t["logprob"] for t in group) / len(group),
                        "score": round(word_score(word_spans), 4),
                    }
                )

            collapsed = [w for w in chunk_words if w["end"] - w["start"] <= 0]
            chunk_dur = record["end"] - record["start"] + 2 * overlap
            tight = [w for w in chunk_words if w["end"] - w["start"] < 0.05 * chunk_dur / max(1, len(chunk_words))]
            if collapsed or (len(chunk_words) >= 10 and len(tight) > 0.3 * len(chunk_words)):
                degenerate.append((where, f"{len(collapsed)} zero-length, {len(tight)} collapsed words"))
            per_chunk_words.append(chunk_words)

        # stitching: overlap margins mean neighbouring chunks can both transcribe a
        # boundary word; cut at the midpoint of the inter-chunk gap so every instant
        # belongs to exactly one chunk (equivalent intent to spec 6.2's "drop
        # overlap-region words from the later chunk", but symmetric and lossless)
        out_records = []
        stitched_words = []
        for i, (record, chunk_words) in enumerate(zip(records, per_chunk_words, strict=True)):
            lower = -1e9 if i == 0 else (records[i - 1]["end"] + record["start"]) / 2
            upper = 1e9 if i == len(records) - 1 else (record["end"] + records[i + 1]["start"]) / 2
            for w in chunk_words:
                mid = (w["start"] + w["end"]) / 2
                w["stitched"] = lower <= mid < upper
                if w["stitched"]:
                    stitched_words.append(w["word"])
            out_records.append(
                {"doc_id": doc_id, "chunk_id": record["chunk_id"],
                 "aligner": chunk_labels[record["chunk_id"]], "words": chunk_words}
            )
            n_words_total += len(chunk_words)

        write_jsonl(words_path, out_records)
        text_path.write_text(" ".join(stitched_words) + "\n", encoding="utf-8")

    print(f"aligned {n_words_total} words across {len(docs)} docs; {len(degenerate)} chunks flagged")
    if unalignable:
        print(f"warning: {len(unalignable)} chunks had CTC-infeasible 1-bests (decode loops), excluded from transcripts:", file=sys.stderr)
        for where, why in unalignable:
            print(f"  {where}: {why}", file=sys.stderr)
    if degenerate:
        for where, why in degenerate[:20]:
            print(f"  {where}: {why}", file=sys.stderr)
        raise RuntimeError(
            f"{len(degenerate)} chunks need aligner escalation and the whisperx/whisper_dtw "
            f"rungs are not implemented; see spec 07 section 6.4"
        )
