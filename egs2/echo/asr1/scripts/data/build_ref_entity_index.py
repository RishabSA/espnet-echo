import argparse
import json
import os
from pathlib import Path

import spacy
from tqdm import tqdm

from scripts.common.io import read_jsonl
from scripts.common.ref_index import entities_payload, mentions_to_entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0b: reference entity index via spaCy NER projection, for corpora without native entity tags (AMI, TED-LIUM; spec 07 section 5.3, source spacy_projected). Lowercased references (TED-LIUM) degrade NER recall; the index is still usable but say so when reporting.")
    parser.add_argument("--manifest", type=str, required=True, help="Corpus manifest whose ref_path files get tagged (required).")
    parser.add_argument("--out-dir", type=str, required=True, help="Output dir for ref_entities/<doc>.json (required).")
    parser.add_argument("--times-dir", type=str, default=None, help="Optional dir of <doc>.json word-time lists aligned to the reference tokens, attaching speaker and times to occurrences (default: None).")
    parser.add_argument("--model", type=str, default="en_core_web_lg", help="spaCy model; en_core_web_trf is incompatible with the pinned transformers 5.x, docs/06 (default: en_core_web_lg).")
    parser.add_argument("--force", action="store_true", help="Rewrite docs whose index already exists (default: False).")
    args = parser.parse_args()

    nlp = spacy.load(args.model, disable=["lemmatizer"])
    os.makedirs(args.out_dir, exist_ok=True)

    n_entities = n_occ = 0
    for m in tqdm(sorted(read_jsonl(args.manifest), key=lambda d: d["doc_id"]), desc="tagging refs"):
        out_path = Path(args.out_dir) / f"{m['doc_id']}.json"
        if out_path.exists() and not args.force:
            continue
        text = Path(m["ref_path"]).read_text(encoding="utf-8").strip()
        tokens = text.split()
        nlp.max_length = max(nlp.max_length, len(text) + 1)
        doc = nlp(text)
        word_meta = None
        if args.times_dir:
            word_meta = json.loads((Path(args.times_dir) / f"{m['doc_id']}.json").read_text(encoding="utf-8"))
            if len(word_meta) != len(tokens):
                raise ValueError(f"{m['doc_id']}: {len(word_meta)} word-time rows vs {len(tokens)} reference tokens")
        entities = mentions_to_entities(tokens, [(e.label_, e.start_char, e.end_char) for e in doc.ents], word_meta)
        out_path.write_text(json.dumps(entities_payload(m["doc_id"], "spacy_projected", entities), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n_entities += len(entities)
        n_occ += sum(len(e["occurrences"]) for e in entities)

    print(f"tagged with {args.model}: {n_entities} entities, {n_occ} occurrences -> {args.out_dir}")
