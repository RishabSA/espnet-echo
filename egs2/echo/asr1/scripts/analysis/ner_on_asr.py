import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from tqdm import tqdm

from scripts.analysis.nc_distribution import named_categories
from scripts.common.align import DocAlignment, align_doc
from scripts.common.io import append_config, git_sha, read_hyp_text, read_jsonl
from scripts.common.ref_index import span_to_tokens, token_starts
from scripts.mine.mine_candidates import ner_mentions

comparisons = ["hyp_vs_gold", "ref_vs_gold", "hyp_vs_ref_spacy"]


def tag_words(nlp: spacy.language.Language, tokens: list[str]) -> list[tuple[str, frozenset]]:
    starts = token_starts(tokens)
    out = []
    for label, cs, ce in ner_mentions(nlp, " ".join(tokens)):
        a, b = span_to_tokens(starts, cs, ce)
        out.append((label, frozenset(range(a, b + 1))))
    return out


def hyp_to_ref_words(alignment: DocAlignment) -> dict[int, set[int]]:
    out = defaultdict(set)
    for i, h in enumerate(alignment.ref_to_hyp):
        if h is not None:
            out[alignment.hyp[h].word_idx].add(alignment.ref[i].word_idx)
    return out


def project(mentions: list[tuple[str, frozenset]], h2r: dict[int, set[int]]) -> list[tuple[str, frozenset]]:
    # hypothesis-side mentions expressed in reference word indices; a mention aligned to nothing
    # (a pure insertion) keeps an empty span and can only count as a false positive
    return [(label, frozenset(w for i in words for w in h2r.get(i, ()))) for label, words in mentions]


def prf_counts(pred: list[tuple[str, frozenset]], gold: list[tuple[str, frozenset]]) -> Counter:
    # lenient matching: same label and at least one shared reference word
    c = Counter()
    for label, words in pred:
        c[("pred", label)] += 1
        c[("tp_pred", label)] += any(gl == label and words & gw for gl, gw in gold)
    for label, words in gold:
        c[("gold", label)] += 1
        c[("tp_gold", label)] += any(pl == label and words & pw for pl, pw in pred)
    return c


def prf(c: Counter, labels: list[str]) -> dict:
    def block(ls: list[str]) -> dict:
        pred, tp_p = sum(c[("pred", l)] for l in ls), sum(c[("tp_pred", l)] for l in ls)
        gold, tp_g = sum(c[("gold", l)] for l in ls), sum(c[("tp_gold", l)] for l in ls)
        p, r = (tp_p / pred if pred else None), (tp_g / gold if gold else None)
        f = 2 * p * r / (p + r) if p and r else None
        return {"precision": p, "recall": r, "f1": f, "n_pred": pred, "n_gold": gold}

    return {"all": block(labels), **{l: block([l]) for l in labels}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4.2 / Analysis 9: NER-on-ASR degradation. spaCy over the stitched pass-1 transcript and over the reference, both scored against the gold entity index and against each other, lenient span matching; writes <run-dir>/metrics/ner_on_asr_<split>.json.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir with pass1 transcripts (required).")
    parser.add_argument("--refs-dir", type=str, default="data/derived/earnings21-conec/refs", help="Reference transcripts (default: data/derived/earnings21-conec/refs).")
    parser.add_argument("--ref-entities", type=str, default="data/derived/earnings21-conec/ref_entities", help="Gold entity index dir (default: data/derived/earnings21-conec/ref_entities).")
    parser.add_argument("--manifest", type=str, default="data/derived/earnings21-conec/manifest.jsonl", help="Corpus manifest for split membership (default: data/derived/earnings21-conec/manifest.jsonl).")
    parser.add_argument("--split", type=str, default="dev", help="Split to analyse, or all (default: dev).")
    parser.add_argument("--model", type=str, default="en_core_web_lg", help="spaCy model (default: en_core_web_lg).")
    args = parser.parse_args()

    docs = sorted(m["doc_id"] for m in read_jsonl(args.manifest) if args.split in (m["split"], "all"))
    if not docs:
        raise ValueError(f"no docs in split {args.split!r} of {args.manifest}")
    nlp = spacy.load(args.model, disable=["lemmatizer"])
    labels = sorted(named_categories)

    totals = {k: Counter() for k in comparisons}
    per_doc = []
    for doc in tqdm(docs, desc="tagging"):
        ref_text = (Path(args.refs_dir) / f"{doc}.txt").read_text(encoding="utf-8")
        hyp_text = read_hyp_text(args.run_dir, "pass1", doc)
        alignment = align_doc(ref_text, hyp_text)
        entities = json.loads((Path(args.ref_entities) / f"{doc}.json").read_text(encoding="utf-8"))["entities"]
        gold = [(e["category"], frozenset(range(o["ref_word_span"][0], o["ref_word_span"][1] + 1)))
                for e in entities if e["category"] in named_categories for o in e["occurrences"]]
        ref_spacy = tag_words(nlp, ref_text.split())
        hyp_spacy = project(tag_words(nlp, hyp_text.split()), hyp_to_ref_words(alignment))
        counts = {"hyp_vs_gold": prf_counts(hyp_spacy, gold), "ref_vs_gold": prf_counts(ref_spacy, gold), "hyp_vs_ref_spacy": prf_counts(hyp_spacy, ref_spacy)}
        for k, c in counts.items():
            totals[k].update(c)
        per_doc.append({"doc_id": doc, **{k: prf(c, labels)["all"] for k, c in counts.items()}})

    result = {k: prf(c, labels) for k, c in totals.items()}
    out = Path(args.run_dir) / "metrics" / f"ner_on_asr_{args.split}.json"
    os.makedirs(out.parent, exist_ok=True)
    sha, dirty = git_sha()
    out.write_text(json.dumps({"meta": {"split": args.split, "n_docs": len(docs), "model": args.model, "git_sha": sha, "dirty": dirty, "argv": vars(args)},
                               "metrics": result, "per_doc": per_doc}, indent=2) + "\n", encoding="utf-8")
    append_config(args.run_dir, f"ner_on_asr_{args.split}", {"argv": vars(args), "out": str(out)})

    def cell(v: float | None) -> str:
        return "-" if v is None else f"{v:.3f}"

    print(f"{len(docs)} docs, lenient span match, named categories only")
    print("| comparison | P | R | F1 | n_pred | n_gold |")
    print("|---|---|---|---|---|---|")
    for k in comparisons:
        b = result[k]["all"]
        print(f"| {k} | {cell(b['precision'])} | {cell(b['recall'])} | {cell(b['f1'])} | {b['n_pred']} | {b['n_gold']} |")
    print("| category | hyp_vs_gold F1 | ref_vs_gold F1 | hyp_vs_ref_spacy F1 | n_gold |")
    print("|---|---|---|---|---|")
    for l in labels:
        print(f"| {l} | " + " | ".join(cell(result[k][l]["f1"]) for k in comparisons) + f" | {result['hyp_vs_gold'][l]['n_gold']} |")
