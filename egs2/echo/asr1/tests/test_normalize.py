import pytest

from scripts.common.normalize import normalize

fold_all_cases = [
    ("Kai-Fu Lee", "kai fu lee"),
    ("O'Brien", "obrien"),
    ("  double   spaces ", "double spaces"),
    ("Q1's GAAP figures!", "q1s gaap figures"),
    ("Zelmark,", "zelmark"),
    ("co-founder", "co founder"),
    ("naïve café", "naïve café"),
    ("Kowalski", "kowalski"),
    ("A.B.C.", "abc"),
    ("Vertex (NASDAQ: VRTX)", "vertex nasdaq vrtx"),
    ("", ""),
    ("---", ""),
]

openasr_cases = [
    ("Hello, World!", "hello world"),
    ("Mr. Kowalski", "mister kowalski"),
    ("  spaced   out  ", "spaced out"),
    ("uh well um yes", "well yes"),
    ("100%", "100%"),
    ("It's fine", "it is fine"),
    ("won't", "will not"),
    ("the co-founder", "the co founder"),
]


@pytest.mark.parametrize("raw,expected", fold_all_cases)
def test_fold_all(raw, expected):
    assert normalize(raw, "fold_all") == expected


@pytest.mark.parametrize("raw,expected", openasr_cases)
def test_openasr(raw, expected):
    assert normalize(raw, "openasr") == expected


def test_surface_is_identity():
    assert normalize("  Exactly As-Is!  ", "surface") == "  Exactly As-Is!  "


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        normalize("x", "lower")
