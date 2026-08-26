import numpy as np
import pytest

from scripts.analysis.decompose_inconsistency import (
    build_pairs,
    fit_mixed_model,
    flip_summary,
    flip_table,
    systematic_share,
    window_realizations,
)
from scripts.analysis.e1_windows import context_windows, sample_targets, shift_windows
from scripts.common.align import align_doc

chunks = [{"chunk_id": 0, "start": 0.0, "end": 10.0}, {"chunk_id": 1, "start": 10.0, "end": 20.0}, {"chunk_id": 2, "start": 20.0, "end": 30.0}]
targets = [
    {"uid": "d/a", "occ_id": "a", "doc_id": "d", "entity_id": "e1", "category": "ORG", "n_realized": 2, "t_start": 3.0, "t_end": 4.0, "chunk_id": 0, "position": 0.35, "speaker": "0", "key": "acme", "ref_key": "acme", "correct": True, "entity_consistent": True, "ref_word_span": [2, 2]},
    {"uid": "d/b", "occ_id": "b", "doc_id": "d", "entity_id": "e1", "category": "ORG", "n_realized": 2, "t_start": 18.0, "t_end": 19.0, "chunk_id": 1, "position": 0.85, "speaker": "1", "key": "acme", "ref_key": "acme", "correct": True, "entity_consistent": True, "ref_word_span": [9, 9]},
    {"uid": "d/c", "occ_id": "c", "doc_id": "d", "entity_id": "e2", "category": "PERSON", "n_realized": 2, "t_start": 12.0, "t_end": 12.5, "chunk_id": 1, "position": 0.2, "speaker": "0", "key": "brett", "ref_key": "bret", "correct": False, "entity_consistent": False, "ref_word_span": [6, 6]},
    {"uid": "d/e", "occ_id": "e", "doc_id": "d", "entity_id": "e2", "category": "PERSON", "n_realized": 2, "t_start": 25.0, "t_end": 25.5, "chunk_id": 2, "position": 0.55, "speaker": "0", "key": "bret", "ref_key": "bret", "correct": True, "entity_consistent": False, "ref_word_span": [14, 14]},
]


def test_shift_windows_and_coverage():
    base = shift_windows(chunks, targets, 0.0, 30.0)
    assert [w["source_chunk_id"] for w in base] == [0, 1, 2] and [w["chunk_id"] for w in base] == [0, 1, 2]
    assert {o for w in base for o in w["occ_ids"]} == {"d/a", "d/b", "d/c", "d/e"}
    shifted = shift_windows(chunks, targets, 5.0, 30.0)
    # a (3-4 s) falls outside every shifted window; c moves into chunk 0's window, e into chunk 2's
    assert [(w["start"], w["end"], w["occ_ids"]) for w in shifted] == [(5.0, 15.0, ["d/c"]), (15.0, 25.0, ["d/b"]), (25.0, 30.0, ["d/e"])]


def test_context_windows_cap_and_prompt():
    words = [{"word": "hello", "start": 0.5, "end": 1.0}, {"word": "there", "start": 1.0, "end": 1.5}, {"word": "late", "start": 20.0, "end": 20.5}]
    w = context_windows([targets[1]], 15.0, 40.0, 29.5, words, 100)
    assert w[0]["start"] == pytest.approx(18.0 - 14.25) and w[0]["end"] == pytest.approx(19.0 + 14.25)
    assert w[0]["prompt"] == "hello there"
    span = context_windows([targets[0]], 0.2, 30.0, 29.5, words, 0)
    assert (span[0]["start"], span[0]["end"]) == (2.8, 4.2) and "prompt" not in span[0]
    # a window starting at 0 has no preceding words, so no prompt key
    assert "prompt" not in context_windows([targets[0]], 5.0, 30.0, 29.5, words, 100)[0]


def test_sample_targets_strata():
    picked = sample_targets(targets, 2, 42)
    assert len(picked) == 2 and len(picked & {"d/c", "d/e"}) == 1 and len(picked & {"d/a", "d/b"}) == 1
    assert sample_targets(targets, 10, 42) == {"d/a", "d/b", "d/c", "d/e"}


def test_build_pairs():
    pairs = build_pairs(targets)
    assert len(pairs) == 2
    by = {(p["occ_i"], p["occ_j"]): p for p in pairs}
    assert by[("d/a", "d/b")]["cross_speaker"] and by[("d/a", "d/b")]["cross_chunk"] and not by[("d/a", "d/b")]["differ"]
    assert by[("d/c", "d/e")]["differ"] and not by[("d/c", "d/e")]["cross_speaker"] and by[("d/c", "d/e")]["pos_diff"] == pytest.approx(0.35)


def test_window_realizations_localizes_reference():
    ref = "we met kai fu lee today and later kai fu lee left early"
    base_hyp = "we met Kai Fu Lee today and later Kai Fu Lee left early"
    base = align_doc(ref, base_hyp)
    words = [{"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate(base_hyp.split())]
    target = {"uid": "x", "ref_word_span": [8, 10]}
    window = {"start": 6.0, "end": 12.0, "occ_ids": ["x"]}
    keys = window_realizations(base, words, ref.split(), window, "and later Kai Fu Li left early", [target])
    assert keys == {"x": "kai fu li"}
    # a window that does not reach the occurrence yields no realization
    assert window_realizations(base, words, ref.split(), {"start": 0.0, "end": 3.0}, "we met Kai", [target]) == {"x": None}


def test_flip_and_systematic_math():
    by_occ = {t["uid"]: t for t in targets}
    base_keys = {"d/a": "acme", "d/b": "acme", "d/c": "brett", "d/e": "bret"}
    cond_keys = {"d/a": "acme", "d/b": "acmi", "d/c": "bret", "d/e": None}
    rows = flip_table(base_keys, cond_keys, by_occ)
    s = flip_summary(rows, ["d"], 50, 42)
    assert s["n_paired"] == 3 and s["flip"]["point"] == pytest.approx(2 / 3)
    assert s["repair"]["point"] == pytest.approx(1 / 3) and s["damage"]["point"] == pytest.approx(1 / 3)
    share = systematic_share(targets, {"d/a": False, "d/b": False, "d/c": True, "d/e": False}, ["d"], 50, 42)
    assert share["n_entities_with_error"] == 1 and share["share"]["point"] == 0.0
    assert systematic_share(targets, {"d/a": False, "d/b": False, "d/c": True, "d/e": True}, ["d"], 50, 42)["share"]["point"] == 1.0


def test_mixed_model_recovers_synthetic_effects():
    rng = np.random.default_rng(42)
    pairs = []
    for d in range(40):
        u_doc = rng.normal(0, 0.3)
        for e in range(12):
            u_ent = rng.normal(0, 0.3)
            for _ in range(6):
                xs, xc, pos = rng.integers(0, 2), rng.integers(0, 2), rng.uniform(0, 1)
                logit = -1.5 + 1.0 * xs + 0.6 * xc + 0.0 * pos + u_doc + u_ent
                pairs.append({"doc_id": f"d{d}", "entity_id": f"e{e}", "differ": rng.uniform() < 1 / (1 + np.exp(-logit)), "cross_speaker": bool(xs), "cross_chunk": bool(xc), "pos_diff": pos})
    fit = fit_mixed_model(pairs)
    fe = fit["fixed_effects"]
    assert 0.6 < fe["cross_speaker"]["mean"] < 1.4
    assert 0.2 < fe["cross_chunk"]["mean"] < 1.0
    assert abs(fe["pos_diff"]["mean"]) < 0.4
