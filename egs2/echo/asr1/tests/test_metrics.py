import json
from pathlib import Path

import pytest

from scripts.common.align import align_doc
from scripts.eval.evaluate import alignable_entities, evaluate, load_bias_list

fixture_dir = Path("tests/fixtures/tiny_doc")
expected = json.loads((fixture_dir / "expected.json").read_text())
docs = ["d1", "d2", "d3"]
tol = 1e-9


def _run(phase: str, bias: str | None = None, compare: bool = False, baseline: bool = False) -> dict:
    return evaluate(
        fixture_dir, phase, fixture_dir / "refs", fixture_dir / "ref_entities", docs,
        load_bias_list(fixture_dir / bias) if bias else None,
        (fixture_dir, "pass1") if compare else None,
        (fixture_dir, "pass1") if baseline else None,
    )


@pytest.mark.parametrize("phase", ["pass1", "pass2"])
def test_wer(phase):
    r = _run(phase)
    for d in r["per_doc"]:
        assert d["wer_n_ref"] == expected["n_ref_words"][d["doc_id"]]
        assert d["wer_errors"] / d["wer_n_ref"] == pytest.approx(expected["wer_openasr"][phase][d["doc_id"]], abs=tol)
    assert r["summary"]["wer"] == pytest.approx(expected["wer_openasr"][phase]["corpus"], abs=tol)


@pytest.mark.parametrize("phase", ["pass1", "pass2"])
def test_consistency_and_oracle(phase):
    s = _run(phase)["summary"]
    for k, v in expected["consistency"][phase].items():
        assert s["consistency"][k] == pytest.approx(v, abs=tol), k
    assert s["consistency"]["n_eligible_entities"] == len(expected["consistency"]["eligible_entities"])
    assert s["consistency"]["n_deleted_occurrences"] == expected["deletions"][f"deleted_ref_occurrences_{phase}"]
    assert s["oracle"]["n_realized_occurrences_eligible"] == expected["oracle"]["n_realized_occurrences_eligible"]
    # the frozen oracle numbers describe pass 1; the oracle draws on the evaluated run's own pool
    if phase == "pass1":
        assert s["oracle"]["pass_entity_occ_correct"] == pytest.approx(expected["oracle"]["pass1_entity_occ_correct"], abs=tol)
        assert s["oracle"]["oracle_entity_occ_correct"] == pytest.approx(expected["oracle"]["oracle_entity_occ_correct"], abs=tol)


@pytest.mark.parametrize("phase", ["pass1", "pass2"])
def test_entity_wer_and_retrieval(phase):
    s = _run(phase)["summary"]
    e = expected["entity_wer"]
    assert s["entity_wer"]["n_ref_entity_words"] == e["n_ref_entity_words"]
    assert s["entity_wer"]["overall"] == pytest.approx(e[phase]["overall"], abs=tol)
    assert s["entity_wer"]["oracle"] == pytest.approx(e[phase]["oracle"], abs=tol)
    for cat in ("PERSON", "ORG"):
        assert s["entity_wer"]["per_category"][cat] == pytest.approx(e[phase][cat], abs=tol), cat
    assert s["retrieval_recall"] == pytest.approx(expected["retrieval_recall"][phase], abs=tol)


def test_transitions():
    s = _run("pass2", compare=True)["summary"]["transitions"]
    for k, v in expected["transitions"].items():
        assert s[k] == pytest.approx(v, abs=tol), k


@pytest.mark.parametrize("phase", ["pass1", "pass2"])
@pytest.mark.parametrize("lst", ["oracle", "corrupted"])
def test_bias(phase, lst):
    s = _run(phase, bias=f"bias_{lst}.txt")["summary"]["bias"]
    e = expected["bias"][f"{lst}_list"]
    assert s["n_biased_ref_words"] == e["n_biased_ref_words"]
    for k in ("b_wer", "u_wer", "baer"):
        assert s[k] == pytest.approx(e[phase][k], abs=tol), k
    assert s["n_in_list"] == e[phase]["n_in_list"]
    assert s["bias_insertions"] == e[phase]["insertions"]


def test_amplification():
    s = _run("pass2", bias="bias_corrupted.txt", baseline=True)["summary"]["bias"]
    assert s["amplification"] == pytest.approx(expected["bias"]["corrupted_list"]["amplification_pass2_vs_pass1"], abs=tol)
    # the oracle list has no corrupted entry, so there is nothing to amplify
    assert _run("pass2", bias="bias_oracle.txt", baseline=True)["summary"]["bias"]["amplification"] == 1.0


def test_unalignable_reference_occurrences_are_dropped():
    alignment = align_doc("we met <inaudible> and Kowalski + Nagel", "we met and Kowalski Nagel")
    entities = [
        {"entity_id": "x", "category": "ORG", "canonical_surface": "<inaudible>", "occurrences": [{"occ_id": "x#0", "ref_word_span": [2, 2], "surface": "<inaudible>"}]},
        {"entity_id": "y", "category": "ORG", "canonical_surface": "Kowalski + Nagel", "occurrences": [
            {"occ_id": "y#0", "ref_word_span": [4, 6], "surface": "Kowalski + Nagel"}, {"occ_id": "y#1", "ref_word_span": [5, 5], "surface": "+"}]},
    ]
    kept, dropped = alignable_entities(alignment, entities)
    assert dropped == 2
    assert [(e["entity_id"], [o["occ_id"] for o in e["occurrences"]]) for e in kept] == [("y", ["y#0"])]
