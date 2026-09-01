#!/bin/bash
#SBATCH --job-name=prep-e22
#SBATCH --partition=cpu
#SBATCH --account=bbjs-delta-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=30000M
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

# Earnings-22 preparation (mentor directive 5). The public E22 release has no named-entity
# tags (docs/06 entry 24), so ref_entities comes from spaCy projection; native normalization
# tags land in ref_entities_nlp. VAD over 119 h is the long pole (~14 h at cluster speed);
# segment_vad resumes per doc, so a resubmit continues.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
export XDG_CACHE_HOME="$work_root/cache/xdg"

cd data/raw/speech-datasets && git sparse-checkout add earnings22 && cd "$SLURM_SUBMIT_DIR"
uv run python scripts/data/prepare_earnings.py --raw-dir data/raw/speech-datasets/earnings22 \
  --out-dir data/derived/earnings22 --corpus earnings22 --allow-unmatched
uv run python scripts/data/build_ref_entity_index.py --manifest data/derived/earnings22/manifest.jsonl \
  --out-dir data/derived/earnings22/ref_entities --times-dir data/derived/earnings22/ref_words
uv run python scripts/data/segment_vad.py --manifest data/derived/earnings22/manifest.jsonl --out-dir data/derived/earnings22/vad
