import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from wordfreq import zipf_frequency

from scripts.common.distance import d_lev_norm, d_phon
from scripts.common.io import append_config, read_jsonl
from scripts.common.phonetics import phones


def is_common(norm: str, zipf_max: float) -> bool:
    return all(zipf_frequency(w, "en") >= zipf_max for w in norm.split())


def distance_matrices(variants: list[str], zipf_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # string distance over whitespace-stripped forms so "kai fu lee" and "kaifu lee" collide
    # (hyphens are already folded to spaces by fold_all); phones come from the spaced form
    n = len(variants)
    lev, phon = np.zeros((n, n)), np.zeros((n, n))
    squashed = [v.replace(" ", "") for v in variants]
    for i in range(n):
        for j in range(i + 1, n):
            lev[i, j] = lev[j, i] = d_lev_norm(squashed[i], squashed[j])
            phon[i, j] = phon[j, i] = d_phon(variants[i], variants[j])
    common = np.array([is_common(v, zipf_max) for v in variants], dtype=bool)
    return lev, phon, common[:, None] & common[None, :]


def agglomerate(d: np.ndarray, forbidden: np.ndarray, tau: float) -> list[list[int]]:
    # average linkage with Lance-Williams updates (docs/02 stage 2); a forbidden pair blocks every
    # cluster merge that would bring its two members together, whatever the average distance says
    members = {i: [i] for i in range(len(d))}
    dist = d.astype(float).copy()
    block = forbidden.copy()
    np.fill_diagonal(dist, np.inf)
    while len(members) > 1:
        masked = np.where(block, np.inf, dist)
        i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)
        if masked[i, j] > tau:
            break
        na, nb = len(members[i]), len(members[j])
        merged = (na * dist[i, :] + nb * dist[j, :]) / (na + nb)
        dist[i, :] = dist[:, i] = merged
        block[i, :] = block[:, i] = block[i, :] | block[j, :]
        dist[i, i] = np.inf
        dist[j, :] = dist[:, j] = np.inf
        block[j, :] = block[:, j] = True
        members[i] += members.pop(j)
    return sorted(members.values(), key=lambda m: m[0])


def plurality_surface(mentions: list[dict]) -> str:
    # most frequent normalized spelling, then its most frequent surface, so casing alone cannot swing it
    top_norm = Counter(m["norm"] for m in mentions).most_common(1)[0][0]
    return Counter(m["surface"] for m in mentions if m["norm"] == top_norm).most_common(1)[0][0]


def build_clusters(doc_id: str, cands: list[dict], lam: float, tau: float, min_occ: int, dphon_cap: float,
                   zipf_max: float, mats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> dict:
    variants = sorted({c["norm"] for c in cands})
    lev, phon, both_common = mats if mats is not None else distance_matrices(variants, zipf_max)
    d = lam * lev + (1.0 - lam) * phon
    groups = agglomerate(d, (phon > dphon_cap) | both_common, tau)

    by_norm = defaultdict(list)
    for c in cands:
        by_norm[c["norm"]].append(c)
    survivors = []
    for g in groups:
        mentions = [m for k in g for m in by_norm[variants[k]]]
        if len(mentions) >= min_occ:
            survivors.append((g, mentions))

    # span-level wins: a mention nested inside a longer mention of a surviving cluster is the
    # component-word fallback that was not needed
    by_start = defaultdict(list)
    max_len = 0
    for _, mentions in survivors:
        for m in mentions:
            a, b = m["word_span"]
            by_start[a].append(b)
            max_len = max(max_len, b - a)

    def nested(m: dict) -> bool:
        a, b = m["word_span"]
        return any(y >= b and y - x > b - a for x in range(max(0, a - max_len), a + 1) for y in by_start[x])

    clusters = []
    for g, mentions in survivors:
        mentions = sorted((m for m in mentions if not nested(m)), key=lambda m: m["word_span"])
        if len(mentions) < min_occ:
            continue
        norms = {m["norm"] for m in mentions}
        idx = [k for k in g if variants[k] in norms]
        clusters.append({
            "first_word": mentions[0]["word_span"][0],
            "occ_ids": [m["occ_id"] for m in mentions], "n_occ": len(mentions),
            "variants": dict(Counter(m["surface"] for m in mentions).most_common()),
            "plurality": plurality_surface(mentions),
            "phone_repr": " ".join(phones(Counter(m["norm"] for m in mentions).most_common(1)[0][0])),
            "max_pairwise_d": float(max((d[i, j] for i in idx for j in idx if i < j), default=0.0)),
            "categories": dict(Counter(m["signals"]["ner"] for m in mentions if m["signals"]["ner"]).most_common()),
        })
    clusters.sort(key=lambda c: c.pop("first_word"))
    return {
        "doc_id": doc_id,
        "params": {"lambda": lam, "tau": tau, "min_occ": min_occ, "dphon_cap": dphon_cap, "zipf_max": zipf_max},
        "clusters": [{"cluster_id": f"c_{k:03d}", **c} for k, c in enumerate(clusters)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2b: cluster candidate variant strings with the combined orthographic + phonetic distance under average linkage and the precision guards, attach mentions, write <run-dir>/clusters/<doc>.json (spec 07 sections 5.7, 6.6).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run dir holding <candidates-subdir>/<doc>.jsonl (required).")
    parser.add_argument("--docs", type=str, default="", help="Comma-separated doc ids; empty means every doc with candidates (default: all).")
    parser.add_argument("--candidates-subdir", type=str, default="candidates", help="Input subdir under the run dir (default: candidates).")
    parser.add_argument("--out-subdir", type=str, default="clusters", help="Output subdir under the run dir (default: clusters).")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.25, help="Weight on the string term; 1 - lambda on the phonetic term (default: 0.25).")
    parser.add_argument("--tau", type=float, default=0.10, help="Average-linkage merge threshold on the combined distance (default: 0.10).")
    parser.add_argument("--min-occ", type=int, default=2, help="Drop clusters with fewer mentions (default: 2).")
    parser.add_argument("--dphon-cap", type=float, default=0.25, help="Never merge a variant pair whose phonetic distance exceeds this (default: 0.25).")
    parser.add_argument("--zipf-max", type=float, default=3.5, help="Two variants at or above this Zipf frequency are common words, never merged (default: 3.5).")
    parser.add_argument("--force", action="store_true", help="Rewrite docs whose clusters already exist (default: False).")
    args = parser.parse_args()

    run = Path(args.run_dir)
    docs = args.docs.split(",") if args.docs else sorted(p.stem for p in (run / args.candidates_subdir).glob("*.jsonl"))
    if not docs:
        raise FileNotFoundError(f"no candidate files under {run / args.candidates_subdir}")
    os.makedirs(run / args.out_subdir, exist_ok=True)

    sizes, n_docs = [], 0
    for doc in tqdm(docs, desc="clustering"):
        out_path = run / args.out_subdir / f"{doc}.json"
        if out_path.exists() and not args.force:
            continue
        cands = read_jsonl(run / args.candidates_subdir / f"{doc}.jsonl")
        result = build_clusters(doc, cands, args.lam, args.tau, args.min_occ, args.dphon_cap, args.zipf_max)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        sizes.extend(c["n_occ"] for c in result["clusters"])
        n_docs += 1

    stage = "cluster_variants" if args.out_subdir == "clusters" else f"cluster_variants_{args.out_subdir}"
    summary = {"docs": n_docs, "clusters": len(sizes), "mentions": sum(sizes),
               "median_n_occ": statistics.median(sizes) if sizes else None,
               "share_n_occ_2": sum(s == 2 for s in sizes) / len(sizes) if sizes else None}
    append_config(run, stage, {"argv": vars(args), **summary})
    print(f"{n_docs} docs, {len(sizes)} clusters over {sum(sizes)} mentions; median N_c {summary['median_n_occ']}, "
          f"share at N_c=2 {summary['share_n_occ_2']}")
