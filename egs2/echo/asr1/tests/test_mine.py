import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.e1_windows import stitched_words
from scripts.common.align import align_doc
from scripts.common.io import read_hyp_text
from scripts.mine.cluster_variants import agglomerate, build_clusters, is_common
from scripts.mine.eval_clusters import doc_cluster_metrics
from scripts.mine.mine_candidates import (
    caps_fires,
    load_stoplist,
    mine_doc,
    runs_of,
    sentence_initial_flags,
    word_norm,
)

fixture_dir = Path("tests/fixtures/tiny_doc")
stoplist = load_stoplist("scripts/mine/stoplist_earnings.txt")
mine_defaults = {"zipf_max": 3.5, "conf_max": -0.80, "max_span_words": 4}


def _words(text: str) -> list[dict]:
    return [{"word": w, "start": float(i), "end": i + 0.5, "logprob": -0.1, "chunk_id": 0} for i, w in enumerate(text.split())]


def test_word_rules():
    assert sentence_initial_flags(["Hello", "there.", "Next", "one"]) == [True, False, True, False]
    assert word_norm("Kowalski's,") == "kowalski" and word_norm("Inc.'s") == "inc"
    assert caps_fires("Kowalski", False) and not caps_fires("Kowalski", True)
    assert caps_fires("iPhone", True) and caps_fires("EBITDA", True)
    assert not caps_fires("I", False) and not caps_fires("I'm", False) and not caps_fires("the", False)
    assert runs_of([True, True, True, True, True, False, True], 4) == [(0, 3), (4, 4), (6, 6)]
    assert is_common("market", 3.5) and not is_common("kowalski", 3.5) and not is_common("market kowalski", 3.5)


def test_mine_doc_spans_components_and_stoplist():
    words = _words("Thanks operator we met Kai Fu Lee and GAAP margins rose at Zelmark today")
    words[9]["logprob"] = -1.5
    cands = mine_doc("d", words, [("PERSON", 23, 33)], {"rarity", "ner", "lowconf", "caps"}, stoplist=stoplist, **mine_defaults)
    by_norm = {c["norm"]: c for c in cands}
    span = by_norm["kai fu lee"]
    assert span["word_span"] == [4, 6] and span["occ_id"] == "d#c0#w0004-0006" and span["signals"]["ner"] == "PERSON"
    assert span["surface"] == "Kai Fu Lee" and span["start"] == 4.0 and span["end"] == 6.5
    # component words survive as the fallback and inherit the label
    assert {"kai", "fu", "lee"} <= set(by_norm) and by_norm["fu"]["signals"]["ner"] == "PERSON"
    assert "operator" not in by_norm and "gaap" not in by_norm and "thanks" not in by_norm
    assert by_norm["margins"]["signals"] == {"rarity": False, "ner": None, "lowconf": True, "caps": False}
    assert by_norm["zelmark"]["signals"]["caps"] and by_norm["zelmark"]["signals"]["rarity"]
    assert len({c["occ_id"] for c in cands}) == len(cands)
    # disabling ner removes the span and the components that had no signal of their own
    only_caps = {c["norm"] for c in mine_doc("d", words, [("PERSON", 23, 33)], {"caps"}, stoplist=stoplist, **mine_defaults)}
    assert "kai fu lee" in only_caps and "fu" in only_caps and "margins" not in only_caps


def test_agglomerate_matches_worked_example():
    # docs/10 section 3 distance matrix, average linkage at the documented thresholds
    d = np.array([[0, .063, .125, .422, .863], [.063, 0, .188, .367, .870], [.125, .188, 0, .533, .875],
                  [.422, .367, .533, 0, .881], [.863, .870, .875, .881, 0]])
    free = np.zeros((5, 5), dtype=bool)
    assert agglomerate(d, free, 0.35) == [[0, 1, 2], [3], [4]]
    # single linkage would admit kowalczyk at 0.38 through kowalsky (0.367); average linkage sits at 0.441
    assert agglomerate(d, free, 0.38) == [[0, 1, 2], [3], [4]]
    assert agglomerate(d, free, 0.45) == [[0, 1, 2, 3], [4]]
    veto = free.copy()
    veto[2, 3] = veto[3, 2] = True
    assert agglomerate(d, veto, 0.45) == [[0, 1, 2], [3], [4]]
    assert agglomerate(np.zeros((0, 0)), np.zeros((0, 0), dtype=bool), 0.3) == []
    assert agglomerate(np.zeros((1, 1)), np.zeros((1, 1), dtype=bool), 0.3) == [[0]]


def _fixture_pipeline(tau: float) -> dict:
    result = {}
    for doc in ["d1", "d2", "d3"]:
        cands = mine_doc(doc, stitched_words(fixture_dir, doc), [], {"rarity", "lowconf", "caps"}, stoplist=stoplist, **mine_defaults)
        clusters = build_clusters(doc, cands, 0.5, tau, 2, 0.45, 3.5)
        entities = json.loads((fixture_dir / "ref_entities" / f"{doc}.json").read_text())["entities"]
        alignment = align_doc((fixture_dir / "refs" / f"{doc}.txt").read_text(), read_hyp_text(fixture_dir, "pass1", doc))
        result[doc] = (cands, clusters, *doc_cluster_metrics(doc, cands, clusters, alignment, entities))
    return result


def test_fixture_clusters_and_trap():
    r = _fixture_pipeline(0.35)
    cands, clusters, m, _ = r["d1"]
    norms = {c["norm"] for c in cands}
    assert {"kowalsky", "kowalski", "marcus kowalsky"} <= norms and "operator" not in norms
    kow = next(c for c in clusters["clusters"] if "d1#c0#w0019" in c["occ_ids"])
    assert "d1#c0#w0029" in kow["occ_ids"] and kow["n_occ"] >= 3 and kow["phone_repr"].startswith("K")
    assert m["oracle_list"] == 1 and m["list_hits_any_variant"] == 1 and m["target_occ_covered"] == 3
    # docs/10 section 6, the (a)+A0 disaster case: the vote picks Kowalsky and would overwrite the one correct Kowalski
    assert (m["vote_labeled"], m["vote_correct_now"], m["vote_damage"], m["vote_repair"]) == (3, 1, 1, 0)
    assert (m["oracle_damage"], m["oracle_repair"]) == (0, 2)

    cands, clusters, m, rows = r["d2"]
    assert len(clusters["clusters"]) == 1 and rows[0]["purity"] == 1.0 and not rows[0]["contaminated"]
    assert clusters["clusters"][0]["variants"] == {"Zelmark": 1, "Selmark": 1} and rows[0]["spellings"] == {"zelmark": 2}
    assert (m["vote_damage"], m["vote_repair"]) == (0, 1)

    # the trap: Meridian and Veridian sit at combined distance 0.073, so they merge at the default
    # threshold and contamination is the metric that reports it (fixture README)
    cands, clusters, m, rows = r["d3"]
    assert len(clusters["clusters"]) == 1 and clusters["clusters"][0]["n_occ"] == 3
    assert (m["pure_mentions_named"], m["cluster_mentions_named"], m["contaminated_mentions_named"]) == (2, 3, 3)
    assert rows[0]["spellings"] == {"veridian": 2, "meridian": 1} and rows[0]["named"]
    assert (m["target_occ"], m["target_occ_covered"], m["target_occ_deleted"]) == (3, 3, 1)
    assert m["oracle_list"] == 2 and m["list_hits_any_variant"] == 2 and rows[0]["categories"] == ["ORG"]
    # the merged trap canonicalized by vote would overwrite the correct Meridian
    assert (m["vote_correct_now"], m["vote_damage"], m["vote_repair"]) == (3, 1, 0)
    assert (m["oracle_damage"], m["oracle_repair"]) == (1, 0)


def test_strict_tau_separates_trap():
    _, clusters, m, _ = _fixture_pipeline(0.05)["d3"]
    assert [c["n_occ"] for c in clusters["clusters"]] == [2]
    assert m["contaminated_mentions_named"] == 0 and (m["target_occ_covered"], m["target_occ"]) == (2, 3)


@pytest.mark.slow
def test_spacy_tags_fixture_names():
    import spacy

    from scripts.mine.mine_candidates import ner_mentions

    nlp = spacy.load("en_core_web_lg", disable=["lemmatizer"])
    text = read_hyp_text(fixture_dir, "pass1", "d1")
    assert any("Kowalsk" in text[a:b] for _, a, b in ner_mentions(nlp, text))
