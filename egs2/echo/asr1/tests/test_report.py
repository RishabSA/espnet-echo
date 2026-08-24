import json
from pathlib import Path

from scripts.common.io import read_jsonl
from scripts.eval.evaluate import evaluate, load_bias_list, write_metrics
from scripts.eval.stats import merge_into_summary, run_stats
from scripts.report.aggregate_runs import aggregate
from scripts.report.make_tables import load, render

fixture_dir = Path("tests/fixtures/tiny_doc")
docs = ["d1", "d2", "d3"]


def _evaluate_to(tmp_path: Path, phase: str) -> Path:
    result = evaluate(
        fixture_dir, phase, fixture_dir / "refs", fixture_dir / "ref_entities", docs,
        load_bias_list(fixture_dir / "bias_corrupted.txt"), (fixture_dir, "pass1") if phase == "pass2" else None, (fixture_dir, "pass1"),
    )
    out = tmp_path / "tiny" / "metrics" / f"{phase}_all"
    meta = {"run_name": "tiny", "phase": phase, "split": "all", "corpus": "tiny_doc", "model": None, "n_docs": 3, "docs": docs, "git_sha": "0" * 40, "dirty": False, "argv": {}}
    write_metrics(out, result, meta)
    return out


def test_tables_regenerate_from_summaries(tmp_path):
    p1, p2 = _evaluate_to(tmp_path, "pass1"), _evaluate_to(tmp_path, "pass2")
    merge_into_summary(p1 / "summary.json", run_stats(read_jsonl(p1 / "per_doc.jsonl"), 200, 42), None)
    merge_into_summary(p2 / "summary.json", run_stats(read_jsonl(p2 / "per_doc.jsonl"), 200, 42, read_jsonl(p1 / "per_doc.jsonl")), "tiny/pass1_all")

    s2 = json.loads((p2 / "summary.json").read_text())
    assert s2["metrics"]["consistency"]["ccr"] == 2 / 3 and "n_docs" not in s2["metrics"]
    assert s2["ci"]["ccr"][0] <= 2 / 3 <= s2["ci"]["ccr"][1]
    assert s2["deltas_vs"]["run"] == "tiny/pass1_all" and s2["deltas_vs"]["ccr"]["point"] == 1 / 3

    md = render(load([p1 / "summary.json", p2 / "summary.json"]), floor=load([p1 / "summary.json"])[0])
    assert "| tiny (pass1, all) |" in md and "0.333 [" in md and "0.667 [" in md
    assert "| PERSON |" in md and "## Transitions" in md and "## Biasing-list metrics" in md and "## Paired differences" in md
    assert "(noise floor)" in md

    rows = aggregate(tmp_path)
    assert [r["phase"] for r in rows] == ["pass1", "pass2"]
    assert rows[1]["metrics.transitions.repair"] == 0.25 and "docs" not in rows[0]
