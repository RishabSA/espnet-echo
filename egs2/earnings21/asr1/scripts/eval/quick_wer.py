import argparse
from pathlib import Path

import jiwer

from scripts.common.io import read_jsonl
from scripts.common.normalize import normalize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus WER of stitched pass-1 texts, openasr policy, ratio of sums: the M2 acceptance gate against the published leaderboard number.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir containing pass1/<doc>.txt (required).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21/manifest.jsonl", help="Corpus manifest (default: data/derived/earnings21/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"], help="Which documents to score (default: test).")
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    if args.split != "all":
        manifest = [m for m in manifest if m["split"] == args.split]

    total_err = 0
    total_ref = 0
    for m in sorted(manifest, key=lambda d: d["doc_id"]):
        hyp_path = Path(args.run_dir) / "pass1" / f"{m['doc_id']}.txt"
        ref = normalize(Path(m["ref_path"]).read_text(encoding="utf-8"), "openasr")
        hyp = normalize(hyp_path.read_text(encoding="utf-8"), "openasr")
        out = jiwer.process_words(ref, hyp)
        err = out.substitutions + out.deletions + out.insertions
        n_ref = out.substitutions + out.deletions + out.hits
        total_err += err
        total_ref += n_ref
        print(f"{m['doc_id']}  wer={err / n_ref:.4f}  ref_words={n_ref}")

    print(f"corpus wer ({args.split}, openasr, ratio-of-sums): {total_err / total_ref:.4f}")
