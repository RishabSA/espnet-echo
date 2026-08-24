import re
from functools import lru_cache

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

hyphen_re = re.compile(r"[-‐‑]")
_nonword_re = re.compile(r"[^\w\s]")
_space_re = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _openasr() -> EnglishTextNormalizer:
    # empty spelling map for now: the British-to-American mapping ships with each
    # checkpoint's tokenizer files and gets wired in at M3 once a checkpoint is pinned
    return EnglishTextNormalizer({})


def normalize(text: str, policy: str) -> str:
    if policy == "surface":
        return text
    if policy == "fold_all":
        # lowercase, hyphens to spaces (so Kai-Fu / Kaifu / Kai Fu collide after a
        # whitespace strip downstream), drop punctuation, collapse whitespace
        folded = hyphen_re.sub(" ", text.lower())
        folded = _nonword_re.sub("", folded)
        return _space_re.sub(" ", folded).strip()
    if policy == "openasr":
        return _openasr()(text)
    raise ValueError(f"unknown normalization policy {policy!r}, expected openasr, fold_all, or surface")
