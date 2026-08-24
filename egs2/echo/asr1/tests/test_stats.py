import numpy as np
import pytest

from scripts.eval.stats import bootstrap, paired_wilcoxon, run_stats


def test_identical_docs_zero_width():
    num, den = np.array([2.0, 2.0, 2.0]), np.array([10.0, 10.0, 10.0])
    r = bootstrap(num, den, 200, 42)
    assert r["point"] == 0.2
    assert r["ci95"] == [0.2, 0.2]


def test_two_doc_distribution():
    # resamples of {0/10, 10/10} take values 0, 0.5, 1 with probabilities 1/4, 1/2, 1/4
    num, den = np.array([0.0, 10.0]), np.array([10.0, 10.0])
    r = bootstrap(num, den, 20000, 42)
    assert r["point"] == 0.5
    assert r["ci95"] == [0.0, 1.0]


def test_paired_difference_and_wilcoxon():
    num, den = np.array([1.0, 3.0, 2.0]), np.array([10.0, 10.0, 10.0])
    same = bootstrap(num, den, 200, 42, num, den)
    assert same["point"] == 0.0 and same["ci95"] == [0.0, 0.0]
    assert paired_wilcoxon(num, den, num, den) == 1.0
    worse = num + 1
    d = bootstrap(worse, den, 2000, 42, num, den)
    assert d["point"] == pytest.approx(0.1)
    assert d["ci95"] == [pytest.approx(0.1), pytest.approx(0.1)]
    assert paired_wilcoxon(worse, den, num, den) < 0.5


def test_run_stats_shape():
    docs = [{"doc_id": f"d{i}", "wer_errors": i, "wer_n_ref": 10, "consistent_strict": 1, "consistent_norm": 1, "pairwise_agreement": 1.0,
             "consistent_correct": 1, "eligible_entities": 2, "entity_errors": 1, "oracle_entity_errors": 0, "entity_ref_words": 4,
             "eligible_correct": 3, "oracle_correct": 4, "eligible_realized": 4, "retrieved_mentions": 3, "ref_mentions": 4} for i in range(3)]
    r = run_stats(docs, 100, 42, compare=docs)
    assert r["metrics"]["wer"]["point"] == pytest.approx(0.1)
    assert r["metrics"]["ccr"]["point"] == 0.5
    assert r["metrics"]["wer"]["diff"]["point"] == 0.0
    with pytest.raises(ValueError):
        run_stats(docs, 100, 42, compare=docs[::-1])


def test_zero_denominators():
    assert bootstrap(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 50, 42) == {"point": None, "ci95": None}
    # one empty document must not poison the corpus estimate
    r = bootstrap(np.array([0.0, 2.0, 4.0]), np.array([0.0, 10.0, 10.0]), 2000, 42)
    assert r["point"] == 0.3 and all(np.isfinite(r["ci95"]))
    assert paired_wilcoxon(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])) is None
