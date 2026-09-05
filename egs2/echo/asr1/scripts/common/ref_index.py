from bisect import bisect_right

from scripts.common.normalize import normalize


def token_starts(tokens: list[str]) -> list[int]:
    # character offset of each token in " ".join(tokens)
    starts, pos = [], 0
    for tok in tokens:
        starts.append(pos)
        pos += len(tok) + 1
    return starts


def span_to_tokens(starts: list[int], char_start: int, char_end: int) -> tuple[int, int]:
    # inclusive token span covering the character range [char_start, char_end)
    a = bisect_right(starts, char_start) - 1
    b = bisect_right(starts, max(char_start, char_end - 1)) - 1
    return a, b


def mentions_to_entities(tokens: list[str], mentions: list[tuple[str, int, int]], word_meta: list[dict] | None = None) -> list[dict]:
    # mentions are (category, char_start, char_end) into " ".join(tokens); grouping follows the
    # pinned rule of spec 07 section 5.3: (category, fold_all surface) within the document
    starts = token_starts(tokens)
    groups = {}
    for category, char_start, char_end in mentions:
        a, b = span_to_tokens(starts, char_start, char_end)
        surface = " ".join(tokens[a : b + 1])
        occ = {"span": [a, b], "surface": surface}
        if word_meta is not None:
            occ["speaker"] = word_meta[a].get("speaker")
            if word_meta[a].get("start") is not None and word_meta[b].get("end") is not None:
                occ["start"] = word_meta[a]["start"]
                occ["end"] = word_meta[b]["end"]
        groups.setdefault((category, normalize(surface, "fold_all")), []).append(occ)

    entities = []
    for (category, _), occs in sorted(groups.items(), key=lambda kv: min(o["span"][0] for o in kv[1])):
        occs.sort(key=lambda o: o["span"][0])
        surfaces = [o["surface"] for o in occs]
        entities.append({"category": category, "canonical_surface": max(set(surfaces), key=surfaces.count), "occurrences": occs})
    for k, entity in enumerate(entities):
        entity["entity_id"] = f"e_{k:03d}"
        for j, occ in enumerate(entity["occurrences"]):
            occ["occ_id"] = f"e_{k:03d}#{j}"
    return entities


def entities_payload(doc_id: str, source: str, entities: list[dict]) -> dict:
    return {
        "doc_id": doc_id,
        "source": source,
        "entities": [
            {
                "entity_id": e["entity_id"], "category": e["category"], "canonical_surface": e["canonical_surface"],
                "occurrences": [
                    {"occ_id": o["occ_id"], "ref_word_span": o["span"], "surface": o["surface"],
                     **{k: o[k] for k in ("speaker", "start", "end") if k in o}}
                    for o in e["occurrences"]
                ],
            }
            for e in entities
        ],
    }
