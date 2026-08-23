import ast
from pathlib import Path

from scripts.common.normalize import normalize

nlp_columns = ["token", "speaker", "ts", "endTs", "punctuation", "case", "tags", "wer_tags"]


def _parse_time(field: str) -> float | None:
    # ts/endTs are optional metadata and the corpus contains malformed stamps
    # (a bare "." was seen upstream); anything unparseable means "no timestamp",
    # consistent with how empty fields are treated
    try:
        return float(field)
    except ValueError:
        return None


def parse_nlp(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines[0].split("|") != nlp_columns:
        raise ValueError(f"unexpected .nlp header in {path}: {lines[0]!r}")
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split("|")
        if len(parts) != len(nlp_columns):
            raise ValueError(f"{path}:{n}: expected {len(nlp_columns)} fields, got {len(parts)}: {line!r}")
        # empty tokens and edge whitespace exist in the corpus (3 rows across
        # Earnings-21, annotation artifacts); either would desync token indices
        # from whitespace-split indices, the invariant ref_entities spans rely on
        token = parts[0].strip()
        if not token:
            continue
        if " " in token:
            raise ValueError(f"{path}:{n}: token contains internal whitespace: {parts[0]!r}")
        rows.append(
            {
                "token": token,
                "speaker": parts[1],
                "ts": _parse_time(parts[2]) if parts[2] else None,
                "end_ts": _parse_time(parts[3]) if parts[3] else None,
                "punctuation": parts[4],
                "case": parts[5],
                "wer_tags": ast.literal_eval(parts[7]),
            }
        )
    return rows


def extract_entities(rows: list[dict], tag_types: dict, doc_id: str) -> tuple[list[dict], dict]:
    # wer_tag ids are per-mention span ids, not entity ids (verified on the corpus:
    # every id covers one contiguous span). Entity identity therefore follows the
    # pinned grouping rule of spec 07 section 5.3: (entity_type, fold_all surface)
    # within the document, with no alias merging.
    spans = {}
    for i, row in enumerate(rows):
        for tid in row["wer_tags"]:
            spans.setdefault(tid, []).append(i)

    skipped = sorted(t for t in spans if not t or t not in tag_types)
    noncontig = [t for t, idxs in spans.items() if idxs[-1] - idxs[0] + 1 != len(idxs)]
    if noncontig:
        raise ValueError(f"{doc_id}: non-contiguous wer_tag spans: {noncontig[:5]}")

    groups = {}
    for tid, idxs in spans.items():
        if not tid or tid not in tag_types:
            continue
        category = tag_types[tid]["entity_type"]
        surface = " ".join(rows[i]["token"] for i in idxs)
        key = (category, normalize(surface, "fold_all"))
        occ = {"span": [idxs[0], idxs[-1]], "surface": surface, "speaker": rows[idxs[0]]["speaker"]}
        # word times exist only in some reference variants (ConEC timestamps)
        if rows[idxs[0]]["ts"] is not None and rows[idxs[-1]]["end_ts"] is not None:
            occ["start"] = rows[idxs[0]]["ts"]
            occ["end"] = rows[idxs[-1]]["end_ts"]
        groups.setdefault(key, []).append(occ)

    entities = []
    for (category, _), occs in sorted(groups.items(), key=lambda kv: min(o["span"][0] for o in kv[1])):
        occs.sort(key=lambda o: o["span"][0])
        surfaces = [o["surface"] for o in occs]
        canonical = max(set(surfaces), key=surfaces.count)
        entities.append({"category": category, "canonical_surface": canonical, "occurrences": occs})

    for k, entity in enumerate(entities):
        entity["entity_id"] = f"e_{k:03d}"
        for j, occ in enumerate(entity["occurrences"]):
            occ["occ_id"] = f"e_{k:03d}#{j}"

    stats = {"n_span_ids": len(spans), "n_skipped_tag_ids": len(skipped)}
    return entities, stats
