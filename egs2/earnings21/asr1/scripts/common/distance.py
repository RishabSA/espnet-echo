from functools import lru_cache

from panphon.distance import Distance
from rapidfuzz.distance import Levenshtein

from scripts.common.phonetics import phones, to_ipa


@lru_cache(maxsize=1)
def _panphon() -> Distance:
    return Distance()


def d_lev_norm(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


@lru_cache(maxsize=100000)
def d_phon(a: str, b: str) -> float:
    pa, pb = phones(a), phones(b)
    if not pa and not pb:
        return 0.0
    if not pa or not pb:
        return 1.0
    # feature-based substitution costs keep /b/~/p/ cheaper than /b/~/ʃ/; the
    # salience-weighted variant (weighted_feature_edit_distance_div_maxlen) is a
    # drop-in if the dev sweep wants it
    return min(1.0, _panphon().feature_edit_distance_div_maxlen(to_ipa(pa), to_ipa(pb)))


def combined_distance(a: str, b: str, lam: float = 0.5) -> float:
    return lam * d_lev_norm(a, b) + (1.0 - lam) * d_phon(a, b)
