import argparse
import json
import shutil
import subprocess
from pathlib import Path

fstalign_binary = Path("tools/fstalign/build/fstalign")
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
    parser = argparse.ArgumentParser(description="fstalign WER wrapper (M0.7 stub; per-category entity WER and the jiwer cross-check land at M3).")
    parser.add_argument("--ref", type=str, required=True, help="Reference .nlp or .ctm file (required).")
    parser.add_argument("--hyp", type=str, required=True, help="Hypothesis .txt or .ctm file (required).")
    parser.add_argument("--out", type=str, required=True, help="Path for fstalign's JSON log output (required).")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "binary", "docker"], help="Which fstalign backend to use (default: auto).")
    args = parser.parse_args()

    payload = run_fstalign(Path(args.ref), Path(args.hyp), Path(args.out), backend=args.backend)
    wer = payload.get("wer", {}).get("bestWER", payload.get("wer"))
    print(f"backend={payload['_backend']}")
    print(json.dumps(wer, indent=2) if isinstance(wer, dict) else f"wer={wer}")
