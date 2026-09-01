import argparse
import json
import os
import shutil
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from scripts.common.io import write_jsonl
from scripts.data.prepare_earnings import convert_audio

# appendix-only regression corpus (docs/03): the legacy dev/test talks, not the 452 h train set
split_map = {"dev": "dev", "test": "test"}


def parse_stm(path: str | Path) -> list[dict]:
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(";;"):
            continue
        _file_id, _channel, speaker, start, end, rest = line.split(None, 5)
        _label, _, transcript = rest.partition(">")
        if not rest.startswith("<"):
            transcript = rest
        transcript = transcript.strip()
        if not transcript or transcript == "ignore_time_segment_in_scoring":
            continue
        segments.append({"speaker": speaker, "start": float(start), "end": float(end), "tokens": transcript.split()})
    return sorted(segments, key=lambda s: s["start"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0: TED-LIUM 3 legacy dev/test preparation, one document per talk; entity index comes separately from build_ref_entity_index.py over the lowercased references (docs/03).")
    parser.add_argument("--raw-dir", type=str, default="data/raw/tedlium3/TEDLIUM_release-3", help="Extracted release dir with legacy/<split>/{stm,sph} (default: data/raw/tedlium3/TEDLIUM_release-3).")
    parser.add_argument("--out-dir", type=str, default="data/derived/tedlium3", help="Output dir for derived artifacts (default: data/derived/tedlium3).")
    parser.add_argument("--force", action="store_true", help="Re-encode audio and rewrite outputs even if they exist (default: False).")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it before corpus prep")
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    for d in ["audio", "refs", "ref_words"]:
        os.makedirs(out / d, exist_ok=True)

    manifest = []
    for raw_split, our_split in split_map.items():
        stm_dir = raw / "legacy" / raw_split / "stm"
        talks = sorted(p.stem for p in stm_dir.glob("*.stm"))
        if not talks:
            raise FileNotFoundError(f"no stm files under {stm_dir}")
        for talk in tqdm(talks, desc=f"tedlium {raw_split}"):
            segments = parse_stm(stm_dir / f"{talk}.stm")
            tokens, meta = [], []
            for s in segments:
                for tok in s["tokens"]:
                    tokens.append(tok)
                    # stm has segment times only; every word inherits its segment's span
                    meta.append({"start": s["start"], "end": s["end"], "speaker": s["speaker"]})
            ref_path = out / "refs" / f"{talk}.txt"
            ref_path.write_text(" ".join(tokens) + "\n", encoding="utf-8")
            (out / "ref_words" / f"{talk}.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

            wav_path = out / "audio" / f"{talk}.wav"
            convert_audio(raw / "legacy" / raw_split / "sph" / f"{talk}.sph", wav_path, args.force)
            manifest.append({
                "doc_id": talk, "audio_path": str(wav_path), "duration_s": sf.info(str(wav_path)).duration,
                "ref_path": str(ref_path), "split": our_split, "meta": {"n_segments": len(segments)},
            })

    write_jsonl(out / "manifest.jsonl", manifest)
    n_dev = sum(1 for m in manifest if m["split"] == "dev")
    print(f"{len(manifest)} talks ({n_dev} dev / {len(manifest) - n_dev} test), {sum(m['duration_s'] for m in manifest) / 3600:.2f} h of wav")
    print("next: build_ref_entity_index.py --manifest data/derived/tedlium3/manifest.jsonl --out-dir data/derived/tedlium3/ref_entities --times-dir data/derived/tedlium3/ref_words")
