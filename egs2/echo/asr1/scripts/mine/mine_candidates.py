import argparse
import re
from collections import Counter
from pathlib import Path

import spacy
from tqdm import tqdm
from wordfreq import zipf_frequency

from scripts.analysis.e1_windows import stitched_words
from scripts.common.align import edge_punct_re
from scripts.common.io import append_config, read_hyp_text, write_jsonl
from scripts.common.normalize import normalize
from scripts.common.ref_index import span_to_tokens, token_starts

# spaCy labels the ner signal keeps (spec 07 section 6.5)
ner_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "FAC", "NORP", "LOC", "EVENT", "WORK_OF_ART"}
all_signals = ["rarity", "ner", "lowconf", "caps"]
# English capitalizes the first-person pronoun everywhere, so it carries no proper-noun signal
pronoun_i = {"i", "i'm", "i've", "i'll", "i'd", "i’m", "i’ve", "i’ll", "i’d"}
possessive_re = re.compile(r"['’]s$")
sentence_end_re = re.compile(r"[.?!]['\"”’)]*$")


def clean_word(raw: str) -> str:
    return edge_punct_re.sub("", raw)


def word_norm(raw: str) -> str:
    # possessives fold onto the base spelling so "Kowalski's" clusters with "Kowalski"
    return normalize(possessive_re.sub("", clean_word(raw)), "fold_all")


def is_rare(norm: str, zipf_max: float) -> bool:
    return zipf_frequency(norm, "en") < zipf_max


def caps_fires(raw: str, sentence_initial: bool) -> bool:
    w = clean_word(raw)
    if len(w) < 2 or w.lower() in pronoun_i:
        return False
    if any(ch.isupper() for ch in w[1:]):
        return True
    return w[0].isupper() and not sentence_initial


def sentence_initial_flags(words: list[str]) -> list[bool]:
    return [True] + [bool(sentence_end_re.search(prev)) for prev in words[:-1]]


def runs_of(flags: list[bool], max_len: int) -> list[tuple[int, int]]:
    # maximal runs of firing words, chopped into pieces of at most max_len
    spans, i = [], 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        for a in range(i, j + 1, max_len):
            spans.append((a, min(a + max_len - 1, j)))
        i = j + 1
    return spans


def ner_mentions(nlp: spacy.language.Language, text: str) -> list[tuple[str, int, int]]:
    nlp.max_length = max(nlp.max_length, len(text) + 1)
    return [(e.label_, e.start_char, e.end_char) for e in nlp(text).ents if e.label_ in ner_labels]


def load_stoplist(path: str) -> set[str]:
    if not path:
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {normalize(line, "fold_all") for line in lines if line.strip() and not line.startswith("#")}


def mine_doc(doc_id: str, words: list[dict], mentions: list[tuple[str, int, int]], signals: set[str],
             zipf_max: float, conf_max: float, max_span_words: int, stoplist: set[str]) -> list[dict]:
    raws = [w["word"] for w in words]
    norms = [word_norm(r) for r in raws]
    initial = sentence_initial_flags(raws)
    rare = ["rarity" in signals and bool(n) and is_rare(n, zipf_max) for n in norms]
    caps = ["caps" in signals and caps_fires(r, s) for r, s in zip(raws, initial, strict=True)]
    low = ["lowconf" in signals and w["logprob"] < conf_max for w in words]

    # spans are (a, b) inclusive word indices; NER spans first so their component words inherit
    # the label, then rarity/caps runs, then lone low-confidence words. A component word is only
    # a candidate of its own if some signal fires on it, so "the" in "the Safe Harbor Statement"
    # never becomes a variant
    spans = {}
    if "ner" in signals:
        starts = token_starts(raws)
        for label, cs, ce in mentions:
            spans.setdefault(span_to_tokens(starts, cs, ce), label)
        for (a, b), label in list(spans.items()):
            for i in range(a, b + 1):
                if rare[i] or caps[i] or low[i]:
                    spans.setdefault((i, i), label)
    for a, b in runs_of([r or c for r, c in zip(rare, caps, strict=True)], max_span_words):
        spans.setdefault((a, b), None)
        for i in range(a, b + 1):
            spans.setdefault((i, i), None)
    for i, flag in enumerate(low):
        if flag:
            spans.setdefault((i, i), None)

    cands = []
    for (a, b), label in sorted(spans.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        norm = " ".join(n for n in norms[a : b + 1] if n)
        if len(norm) < 2 or not any(ch.isalpha() for ch in norm) or norm in stoplist:
            continue
        ws = words[a : b + 1]
        cands.append({
            "doc_id": doc_id, "chunk_id": ws[0]["chunk_id"],
            "occ_id": f"{doc_id}#c{ws[0]['chunk_id']}#w{a:04d}" + (f"-{b:04d}" if b > a else ""),
            "word_span": [a, b], "surface": " ".join(clean_word(r) for r in raws[a : b + 1]), "norm": norm,
            "start": ws[0]["start"], "end": ws[-1]["end"], "conf": min(w["logprob"] for w in ws),
            "signals": {"rarity": any(rare[a : b + 1]), "ner": label, "lowconf": any(low[a : b + 1]), "caps": any(caps[a : b + 1])},
        })
    return cands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2a: mine entity/rare-term candidates from the stitched pass-1 transcript, union of four toggleable signals, one record per mention into <run-dir>/candidates (spec 07 sections 5.6, 6.5).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir with pass1/<doc>.txt and pass1/<doc>.words.jsonl (required).")
    parser.add_argument("--docs", type=str, default="", help="Comma-separated doc ids; empty means every doc with a stitched transcript (default: all).")
    parser.add_argument("--signals", type=str, default=",".join(all_signals), help="Comma-separated subset of rarity,ner,lowconf,caps (default: rarity,ner,lowconf,caps).")
    parser.add_argument("--zipf-max", type=float, default=3.5, help="rarity fires below this wordfreq Zipf frequency (default: 3.5).")
    parser.add_argument("--conf-max", type=float, default=-0.80, help="lowconf fires below this mean token logprob (default: -0.80).")
    parser.add_argument("--max-span-words", type=int, default=4, help="Longest rarity/caps run emitted as one candidate; NER spans are never cut (default: 4).")
    parser.add_argument("--stoplist", type=str, default="scripts/mine/stoplist_earnings.txt", help="Boilerplate phrases to drop, one per line; empty string disables (default: scripts/mine/stoplist_earnings.txt).")
    parser.add_argument("--model", type=str, default="en_core_web_lg", help="spaCy model for the ner signal (default: en_core_web_lg).")
    parser.add_argument("--out-subdir", type=str, default="candidates", help="Output subdir under the run dir (default: candidates).")
    parser.add_argument("--force", action="store_true", help="Rewrite docs whose candidates already exist (default: False).")
    args = parser.parse_args()

    signals = set(args.signals.split(","))
    unknown = signals - set(all_signals)
    if unknown:
        raise ValueError(f"unknown signals {sorted(unknown)}; choose from {all_signals}")
    run = Path(args.run_dir)
    docs = args.docs.split(",") if args.docs else sorted(p.name[: -len(".words.jsonl")] for p in (run / "pass1").glob("*.words.jsonl"))
    if not docs:
        raise FileNotFoundError(f"no word timings under {run / 'pass1'}")
    stoplist = load_stoplist(args.stoplist)
    nlp = spacy.load(args.model, disable=["lemmatizer"]) if "ner" in signals else None

    totals = Counter()
    for doc in tqdm(docs, desc="mining"):
        out_path = run / args.out_subdir / f"{doc}.jsonl"
        if out_path.exists() and not args.force:
            continue
        text = read_hyp_text(run, "pass1", doc)
        words = stitched_words(run, doc)
        if len(text.split()) != len(words):
            raise ValueError(f"{doc}: {len(text.split())} transcript tokens vs {len(words)} stitched words")
        mentions = ner_mentions(nlp, text) if nlp else []
        cands = mine_doc(doc, words, mentions, signals, args.zipf_max, args.conf_max, args.max_span_words, stoplist)
        write_jsonl(out_path, cands)
        totals["docs"] += 1
        totals["candidates"] += len(cands)
        totals["norms"] += len({c["norm"] for c in cands})
        totals["words"] += len(words)
        for s in all_signals:
            totals[s] += sum(bool(c["signals"][s]) for c in cands)

    stage = "mine_candidates" if args.out_subdir == "candidates" else f"mine_candidates_{args.out_subdir}"
    append_config(run, stage, {"argv": vars(args), **totals})
    print(f"{totals['docs']} docs, {totals['candidates']} candidates ({totals['norms']} distinct norms) over {totals['words']} words; "
          + ", ".join(f"{s} {totals[s]}" for s in all_signals))
