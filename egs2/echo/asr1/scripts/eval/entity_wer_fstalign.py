import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from scripts.common.align import edge_punct_re
from scripts.common.io import read_hyp_text

fstalign_binary = Path("tools/fstalign/build/fstalign")
# enforced categories: the ones whose tokens are plain words. Numeric and contraction classes
# depend on fstalign's synonym engine, and GPE/LOC/FAC are dominated by dotted initialisms that
# the ConEC references write inconsistently (US vs C.L.), which fstalign scores as errors
enforced_categories = ["PERSON", "ORG", "PRODUCT", "NORP"]
dotted_abbrev_re = re.compile(r"^(\w\.)+\w?\.?$")
docker_image = "revdotcom/fstalign"


def resolve_backend(backend: str) -> str:
    if backend in ("binary", "auto") and fstalign_binary.exists():
        return "binary"
    if backend in ("docker", "auto") and shutil.which("docker"):
        probe = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            return "docker"
        if backend == "docker":
            raise RuntimeError(f"docker requested but the daemon is not running: {probe.stderr.strip()}")
    raise RuntimeError(
        f"no fstalign backend available: build the binary into {fstalign_binary} "
        f"(cmake + OpenFST, see github.com/revdotcom/fstalign) or start docker and pull {docker_image}"
    )


def run_fstalign(ref: Path, hyp: Path, out_json: Path, backend: str = "auto") -> dict:
    resolved = resolve_backend(backend)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if resolved == "binary":
        cmd = [str(fstalign_binary), "wer", "--ref", str(ref), "--hyp", str(hyp),
               "--json-log", str(out_json)]
    else:
        # mount the repo root so container paths mirror host paths
        cwd = Path.cwd()
        cmd = ["docker", "run", "--rm", "-v", f"{cwd}:{cwd}", "-w", str(cwd), docker_image,
               "fstalign", "wer", "--ref", str(ref.resolve()), "--hyp", str(hyp.resolve()),
               "--json-log", str(out_json.resolve())]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"fstalign ({resolved}) failed with code {result.returncode} on ref={ref} hyp={hyp}:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload["_backend"] = resolved
    return payload
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fstalign wrapper: single ref/hyp scoring, or the optional one-shot cross-check of an evaluate.py metrics dir against fstalign's per-category WER over .nlp references (docs/06 entry 22).")
    parser.add_argument("--ref", type=str, default=None, help="Reference .nlp or .ctm file for single-pair scoring (default: None).")
    parser.add_argument("--hyp", type=str, default=None, help="Hypothesis .txt or .ctm file for single-pair scoring (default: None).")
    parser.add_argument("--out", type=str, default=None, help="Path for fstalign's JSON log output in single-pair scoring (default: None).")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "binary", "docker"], help="Which fstalign backend to use (default: auto).")
    parser.add_argument("--check-metrics-dir", type=str, default=None, help="Metrics dir written by evaluate.py to cross-check per document against fstalign (default: None).")
    parser.add_argument("--nlp-dir", type=str, default="data/raw/conec/earnings21/transcripts/timestamps", help="Directory of <doc>.nlp references for the cross-check (default: data/raw/conec/earnings21/transcripts/timestamps).")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Largest absolute divergence tolerated on an enforced category (PERSON, ORG, PRODUCT, NORP) before the check raises (default: 0.03).")
    parser.add_argument("--min-ref-words", type=int, default=100, help="Categories with fewer reference words are reported but not enforced (default: 100).")
    args = parser.parse_args()

    if args.check_metrics_dir:
        metrics_dir = Path(args.check_metrics_dir)
        summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
        run_dir = Path("runs") / summary["run_name"]
        work = metrics_dir / "fstalign"
        os.makedirs(work, exist_ok=True)
        errors, ref_words = Counter(), Counter()
        for doc_id in summary["docs"]:
            # fstalign keeps punctuation inside tokens (inc.'s, c.l.) and resolves contractions,
            # hyphens and numerals through its own synonym engine, so it gets the hypothesis with
            # only boundary punctuation stripped per token; any other normalization fights it
            hyp_path = work / f"{doc_id}.txt"
            words = [w if dotted_abbrev_re.match(w) else edge_punct_re.sub("", w) for w in read_hyp_text(run_dir, summary["phase"], doc_id).split()]
            hyp_path.write_text(" ".join(w for w in words if w) + "\n", encoding="utf-8")
            payload = run_fstalign(Path(args.nlp_dir) / f"{doc_id}.nlp", hyp_path, work / f"{doc_id}.json", backend=args.backend)
            best = payload["wer"]["bestWER"]
            errors["_wer"] += best["numErrors"]
            ref_words["_wer"] += best["numWordsInReference"]
            for cat, v in payload["wer"]["classWER"].items():
                errors[cat] += v["numErrors"]
                ref_words[cat] += v["numWordsInReference"]
            print(f"{doc_id} fstalign wer={best['wer']:.4f}")

        ours = {"_wer": summary["metrics"]["wer"], **summary["metrics"]["entity_wer"]["per_category"]}
        report = {}
        for cat in sorted(errors):
            fst = errors[cat] / ref_words[cat]
            mine = ours.get(cat)
            report[cat] = {"fstalign": fst, "ours": mine, "diff": None if mine is None else mine - fst, "n_ref_words": ref_words[cat]}
            shown = "n/a" if mine is None else f"{mine:.4f} diff={mine - fst:+.4f}"
            print(f"{cat:14s} n={ref_words[cat]:6d} fstalign={fst:.4f} ours={shown}")
        (work / "check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        over = [c for c, r in report.items() if c in enforced_categories and r["diff"] is not None and r["n_ref_words"] >= args.min_ref_words and abs(r["diff"]) > args.tolerance]
        if over:
            raise RuntimeError(f"per-category entity WER diverges from fstalign by more than {args.tolerance} on {over}; details in {work / 'check.json'}")
    else:
        if not (args.ref and args.hyp and args.out):
            raise ValueError("single-pair scoring needs --ref, --hyp, and --out (or use --check-metrics-dir)")
        payload = run_fstalign(Path(args.ref), Path(args.hyp), Path(args.out), backend=args.backend)
        wer = payload.get("wer", {}).get("bestWER", payload.get("wer"))
        print(f"backend={payload['_backend']}")
        print(json.dumps(wer, indent=2) if isinstance(wer, dict) else f"wer={wer}")
