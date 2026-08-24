import re
from dataclasses import dataclass

import jiwer

from scripts.common.normalize import hyphen_re, normalize

# reference transcripts carry <inaudible>/<unk>/<crosstalk>; no ASR output can match them
tag_re = re.compile(r"^<[^>]*>$")
edge_punct_re = re.compile(r"^\W+|\W+$")


@dataclass()
class Token:
    raw: str
    # what preceded this piece in the source text: " " for a whitespace boundary, "-" for a hyphen split
    joiner: str
    key: str
    word_idx: int


@dataclass()
class Realization:
    surface: str
    key: str
    hyp_span: tuple[int, int]


@dataclass()
class DocAlignment:
    ref: list[Token]
    hyp: list[Token]
    ref_to_hyp: list[int | None]
    insert_before: dict[int, list[int]]


def tokenize(text: str) -> list[Token]:
    # hyphenated words split into pieces on both sides so "listen-only" aligns with the
    # reference's "listen only" and vice versa; word_idx keeps ref_word_span indexing intact
    tokens = []
    for word_idx, word in enumerate(text.split()):
        if tag_re.match(word):
            continue
        for k, piece in enumerate(hyphen_re.split(word)):
            key = normalize(piece, "fold_all")
            if key:
                tokens.append(Token(piece, "-" if k else " ", key, word_idx))
    return tokens


def align_doc(ref_text: str, hyp_text: str) -> DocAlignment:
    ref, hyp = tokenize(ref_text), tokenize(hyp_text)
    out = jiwer.process_words(" ".join(t.key for t in ref), " ".join(t.key for t in hyp))
    if len(out.references[0]) != len(ref) or len(out.hypotheses[0]) != len(hyp):
        raise ValueError(f"jiwer retokenized the key stream: ref {len(out.references[0])} vs {len(ref)}, hyp {len(out.hypotheses[0])} vs {len(hyp)}")

    ref_to_hyp = [None] * len(ref)
    insert_before = {}
    for c in out.alignments[0]:
        if c.type in ("equal", "substitute"):
            for r, h in zip(range(c.ref_start_idx, c.ref_end_idx), range(c.hyp_start_idx, c.hyp_end_idx), strict=True):
                ref_to_hyp[r] = h
        elif c.type == "insert":
            insert_before.setdefault(c.ref_start_idx, []).extend(range(c.hyp_start_idx, c.hyp_end_idx))
    return DocAlignment(ref, hyp, ref_to_hyp, insert_before)


def realize(alignment: DocAlignment, ref_word_span: list[int]) -> Realization | None:
    a, b = ref_word_span
    ref_idx = [i for i, t in enumerate(alignment.ref) if a <= t.word_idx <= b]
    if not ref_idx:
        raise ValueError(f"ref_word_span {ref_word_span} covers no alignable reference tokens")

    # span-level: everything aligned to in-span reference tokens plus insertions strictly
    # inside the span; insertions at either boundary belong to the neighbourhood
    hyp_idx = []
    for i in ref_idx:
        if i != ref_idx[0]:
            hyp_idx.extend(alignment.insert_before.get(i, []))
        if alignment.ref_to_hyp[i] is not None:
            hyp_idx.append(alignment.ref_to_hyp[i])
    if not hyp_idx:
        return None

    lo, hi = min(hyp_idx), max(hyp_idx)
    pieces = alignment.hyp[lo : hi + 1]
    surface = pieces[0].raw + "".join(t.joiner + t.raw for t in pieces[1:])
    return Realization(edge_punct_re.sub("", surface), " ".join(t.key for t in pieces), (lo, hi))


def realize_entities(alignment: DocAlignment, entities: list[dict]) -> dict[str, Realization | None]:
    return {o["occ_id"]: realize(alignment, o["ref_word_span"]) for e in entities for o in e["occurrences"]}
