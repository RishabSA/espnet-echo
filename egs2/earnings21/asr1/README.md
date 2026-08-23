# ECHO: Entity Consistency via Hypothesis-derived Occurrences (Earnings-21 recipe)

Long recordings are chunked and decoded independently, so the same named entity gets transcribed inconsistently within one document. ECHO treats the entity lexicon as latent rather than given: candidate entities come from the document's own first pass, from a provided biasing list, or from an external knowledge base, and cross-occurrence acoustic likelihood arbitrates among candidate spellings before a biasing-equipped second pass enforces the result. Companion contributions: within-document entity consistency metrics (ECR/CCR, repair/damage/lock-in transitions) and an analysis of how contextual biasing amplifies errors in imperfect lists.

Research project with CMU WAVLab. Primary data: Earnings-21/22. Models run frozen (Whisper large-v3 via ESPnet's Whisper integration, OWSM later); single-GPU budget.

## Layout

- `scripts/` flat CLI scripts: `common/` (engine, normalization, phonetics, IO), `data/` (corpus prep, VAD), `decode/` (pass 1, word alignment), `eval/`, `analysis/`, `slurm/`
- `tests/` pytest suite incl. the frozen `tiny_doc` metric fixture
- `data/`, `runs/` local artifacts (gitignored); `docs/` full project documentation (local only)

Environment is managed with uv from this directory: `uv sync`, then `uv run python scripts/...`. ESPnet itself is installed editable from the repo root via `[tool.uv.sources]`.
