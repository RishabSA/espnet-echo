# tiny_doc fixture

Hand-built 3-document corpus with hand-computed metric values (spec 07 section 7.5). Frozen 2026-08-09; any metric-definition change updates `expected.json` **by hand first**, then the code. `pass2/` is a hand-written hypothetical enforcement output for metric testing, not the product of running A0.

## Design

| entity | doc | category | ref occurrences | pass1 realizations | pass2 realizations | exercises |
|---|---|---|---|---|---|---|
| e1 Kowalski | d1 | PERSON | 3 (idx 12, 19, 29) | Kowalsky, Kowalski, Kowalsky | Kowalski x3 | **two repairs** (occ 0, occ 2) |
| e2 Zelmark | d2 | ORG | 2 (idx 6, 16) | Zelmark, Selmark | Selmark x2 | **one damage** (occ 0), **one lock-in** (occ 1) |
| e3 Meridian | d3 | ORG | 2 (idx 4, 17) | Meridian, **deleted** | same | **the deletion**: one realized occurrence, so e3 is excluded from the ECR/CCR denominator and counted separately |
| e4 Veridian | d3 | ORG | 2 (idx 10, 23) | Veridian, veridian | same | strict-vs-normalized split: `ecr_strict` sees inconsistency, `fold_all` does not |

Non-entity errors keeping WER honest: d1 analysts->analysis (survives pass 2, enforcement must not touch it), d2 freight->flight.

**Clustering trap**: e3 Meridian and e4 Veridian are genuinely different entities whose combined distance sits under the default merge threshold `tau = 0.35` (see `test_distance.py::test_combined`). A clusterer that merges them manufactures a consistency error; cluster purity (spec 07 section 6.7) is what has to catch it.

## Expected values

`expected.json` holds exact-double values for per-doc and corpus WER (openasr policy, ratio of sums), ECR strict/norm/pairwise, CCR, the transition rates over the 8 occurrences realized in both passes (held 4, repair 2, damage 1, lock-in 1), deletion counts, and the occurrence-level oracle numbers. All values were hand-derived and cross-checked by `gen_fixture.py` (jiwer for WER, direct recomputation for the rest) before freezing. B-WER/U-WER, BAER, and the full oracle-gap entity WER get added by hand at M3 when their implementations land.

## M3 additions (2026-08-24, hand-derived before implementation)

`entity_wer` (errors on reference entity words over the 9 entity words: pass1 4/9 with oracle 1/9 since e3's deletion survives any canonicalizer; pass2 3/9 with oracle 3/9 because the oracle draws on the evaluated run's own pool and pass2 no longer contains Zelmark), `retrieval_recall` (canonical key found as a token subsequence, capped per entity at $N_c$: pass1 5/9, pass2 6/9), and `bias` under two lists in `bias_oracle.txt` / `bias_corrupted.txt`. The corrupted list swaps Kowalsky/Selmark for Kowalski/Zelmark and is the fixture's demonstration of the Le-2021 blind spot: B-WER improves (4/9 to 1/4) while the entity errors move into U-WER (2/109 to 5/114), and BAER (3/6 pass1, 2/5 pass2) is the number that exposes it. The amplification factor pairs pass2 as the biased run against pass1 as the baseline over the 8 realized occurrences: (2/8)/(3/8) = 2/3.

## M4 additions (2026-09-02)

`pass1/<doc>.words.jsonl` gives the miner word timings: one word per token, evenly spaced over the chunk, logprob copied from the token, all stitched. Hand-derived M4 expectations live in `tests/test_mine.py`: at the default `tau = 0.35` the clustering trap merges Meridian and Veridian (combined distance 0.073), which `eval_clusters` reports as contamination 1.0 on d3 with purity 2/3; at `tau = 0.05` they separate, Meridian becomes a singleton and drops, and coverage on d3 falls to 2/3. No frozen metric in `expected.json` depends on these files.
