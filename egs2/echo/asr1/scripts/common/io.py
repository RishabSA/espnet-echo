import json
import os
import subprocess
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def git_sha() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return sha, bool(status)


def create_run_dir(run_dir: str | Path, force: bool = False) -> Path:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(f"run dir {run_dir} is not empty; pass --force to write into it")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def append_config(run_dir: str | Path, stage: str, entry: dict) -> None:
    # per spec section 5.10: one config.json per run, one key per stage, never truncated
    config_path = Path(run_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    sha, dirty = git_sha()
    config[stage] = {**entry, "git_sha": sha, "dirty": dirty}
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_hyp_text(run_dir: str | Path, phase: str, doc_id: str) -> str:
    # real runs carry the midgap-stitched transcript; the fixture has no .txt, so
    # fall back to the chunk 1-bests, which is exact when chunks do not overlap
    text_path = Path(run_dir) / phase / f"{doc_id}.txt"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8")
    records = read_jsonl(Path(run_dir) / phase / f"{doc_id}.jsonl")
    return " ".join(r["hyps"][0]["text"] for r in records)
