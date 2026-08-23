from functools import lru_cache

from g2p_en import G2p

# CMU ARPAbet (stress stripped) to IPA using panphon-known segments only; ER maps to
# two segments because panphon has no rhotacized-vowel entry
arpabet_to_ipa = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "əɹ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


@lru_cache(maxsize=1)
def _g2p() -> G2p:
    return G2p()


@lru_cache(maxsize=100000)
def phones(text: str) -> tuple[str, ...]:
    raw = _g2p()(text)
    return tuple(p.rstrip("012") for p in raw if p.rstrip("012") in arpabet_to_ipa)


def to_ipa(arpabet: tuple[str, ...]) -> str:
    return "".join(arpabet_to_ipa[p] for p in arpabet)
