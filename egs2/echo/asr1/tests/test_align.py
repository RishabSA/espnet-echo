import json
from pathlib import Path

from scripts.common.align import align_doc, realize, realize_entities, tokenize
from scripts.common.io import read_hyp_text

fixture_dir = Path("tests/fixtures/tiny_doc")


def _realizations(doc: str, phase: str) -> dict[str, str | None]:
    ref = (fixture_dir / "refs" / f"{doc}.txt").read_text(encoding="utf-8")
    entities = json.loads((fixture_dir / "ref_entities" / f"{doc}.json").read_text())["entities"]
    alignment = align_doc(ref, read_hyp_text(fixture_dir, phase, doc))
    return {k: (v.surface if v else None) for k, v in realize_entities(alignment, entities).items()}


def test_fixture_pass1():
    assert _realizations("d1", "pass1") == {"e1#0": "Kowalsky", "e1#1": "Kowalski", "e1#2": "Kowalsky"}
    assert _realizations("d2", "pass1") == {"e2#0": "Zelmark", "e2#1": "Selmark"}
    assert _realizations("d3", "pass1") == {"e3#0": "Meridian", "e3#1": None, "e4#0": "Veridian", "e4#1": "veridian"}


def test_fixture_pass2():
    assert _realizations("d1", "pass2") == {"e1#0": "Kowalski", "e1#1": "Kowalski", "e1#2": "Kowalski"}
    assert _realizations("d2", "pass2") == {"e2#0": "Selmark", "e2#1": "Selmark"}
    assert _realizations("d3", "pass2") == {"e3#0": "Meridian", "e3#1": None, "e4#0": "Veridian", "e4#1": "veridian"}


def test_tokenize_pieces():
    tokens = tokenize("COVID-19 listen-only, <inaudible> M&A – the-")
    assert [t.key for t in tokens] == ["covid", "19", "listen", "only", "ma", "the"]
    assert [t.joiner for t in tokens] == [" ", "-", " ", "-", " ", " "]
    assert [t.word_idx for t in tokens] == [0, 0, 1, 1, 3, 5]


def test_multi_token_span_insertions():
    alignment = align_doc("the acme widget corp reported growth", "the acme big widget corp also reported growth")
    r = realize(alignment, [1, 3])
    # the internal insertion is part of the realized span, the boundary insertion is not
    assert r.surface == "acme big widget corp"
    assert r.key == "acme big widget corp"
    assert r.hyp_span == (1, 4)


def test_hyphen_surface_and_edge_punct():
    alignment = align_doc("we met kai fu lee today", "We met Kai-Fu Lee, today.")
    r = realize(alignment, [2, 4])
    assert r.surface == "Kai-Fu Lee"
    assert r.key == "kai fu lee"
    # a hyphenated reference word maps to two tokens that still carry its whitespace index
    alignment = align_doc("during COVID-19 sales fell", "during covid 19 sales fell")
    assert realize(alignment, [1, 1]).surface == "covid 19"


def test_deleted_span():
    alignment = align_doc("the meridian team", "the team")
    assert realize(alignment, [1, 1]) is None
