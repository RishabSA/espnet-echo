import json
from pathlib import Path

from scripts.common.io import append_config, create_run_dir, read_jsonl, write_jsonl

fixture_dir = Path("tests/fixtures/tiny_doc")
docs = ["d1", "d2", "d3"]


def test_jsonl_round_trip(tmp_path):
    records = [
        {"doc_id": "d1", "chunk_id": 0, "nested": {"a": [1, 2.5]}, "text": "ünïcode ɡ"},
        {"doc_id": "d1", "chunk_id": 1, "empty": True},
    ]
    path = tmp_path / "x.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path) == records


def test_run_dir_force_guard(tmp_path):
    run_dir = tmp_path / "run"
    create_run_dir(run_dir)
    (run_dir / "config.json").write_text("{}")
    try:
        create_run_dir(run_dir)
        raise AssertionError("expected FileExistsError on non-empty run dir")
    except FileExistsError:
        pass
    create_run_dir(run_dir, force=True)


def test_append_config_preserves_stages(tmp_path):
    append_config(tmp_path, "pass1", {"argv": ["a"]})
    append_config(tmp_path, "canon", {"argv": ["b"]})
    config = json.loads((tmp_path / "config.json").read_text())
    assert set(config) == {"pass1", "canon"}
    for stage in config.values():
        assert "git_sha" in stage and "dirty" in stage


def test_pass_fixture_schema():
    required = {"rank", "text", "tokens", "sum_logprob", "avg_logprob", "beam_score"}
    for doc in docs:
        for phase in ["pass1", "pass2"]:
            records = read_jsonl(fixture_dir / phase / f"{doc}.jsonl")
            assert [r["chunk_id"] for r in records] == list(range(len(records)))
            for record in records:
                assert record["end"] > record["start"]
                for hyp in record["hyps"]:
                    assert required <= hyp.keys()
                    assert hyp["avg_logprob"] == hyp["sum_logprob"] / len(hyp["tokens"])


def test_ref_entities_spans_match_reference():
    for doc in docs:
        data = json.loads((fixture_dir / "ref_entities" / f"{doc}.json").read_text())
        ref_words = (fixture_dir / "refs" / f"{doc}.txt").read_text().split()
        assert data["doc_id"] == doc
        for entity in data["entities"]:
            for occ in entity["occurrences"]:
                lo, hi = occ["ref_word_span"]
                assert " ".join(ref_words[lo : hi + 1]) == occ["surface"]
