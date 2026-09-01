#!/bin/bash
#SBATCH --job-name=prep-ted3
#SBATCH --partition=cpu
#SBATCH --account=bbjs-delta-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=30000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

# TED-LIUM 3, legacy dev+test talks only (~5 h; appendix regression corpus, docs/03). The
# release tarball is ~51 GB because of the train set; only the legacy eval dirs are extracted.
# spaCy NER on the lowercased references under-recalls; documented in docs/06 entry 24.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
export XDG_CACHE_HOME="$work_root/cache/xdg"

mkdir -p data/raw/tedlium3
if [ ! -d data/raw/tedlium3/TEDLIUM_release-3/legacy/test ]; then
  wget -q "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz" -O data/raw/tedlium3/TEDLIUM_release-3.tgz
  tar -xzf data/raw/tedlium3/TEDLIUM_release-3.tgz -C data/raw/tedlium3 \
    TEDLIUM_release-3/legacy/dev TEDLIUM_release-3/legacy/test
  rm data/raw/tedlium3/TEDLIUM_release-3.tgz
fi
uv run python scripts/data/prepare_tedlium.py
uv run python scripts/data/build_ref_entity_index.py --manifest data/derived/tedlium3/manifest.jsonl \
  --out-dir data/derived/tedlium3/ref_entities --times-dir data/derived/tedlium3/ref_words
uv run python scripts/data/segment_vad.py --manifest data/derived/tedlium3/manifest.jsonl --out-dir data/derived/tedlium3/vad
