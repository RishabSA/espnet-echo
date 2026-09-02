#!/bin/bash
#SBATCH --job-name=prep-ted3
#SBATCH --partition=cpu
#SBATCH --account=bbjs-delta-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=30000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

# TED-LIUM 3, legacy dev+test talks only (~4.3 h; appendix regression corpus, docs/03).
# The corpus was withdrawn from public distribution (OpenSLR resource 51 removed, LIUM page
# and the HF LIUM/tedlium repo gone; verified 2026-09-01, docs/06 entry 24). kfajdsl/tedlium on
# HF is a verbatim file mirror of the original LIUM/tedlium repo: its legacy dev/test tarballs
# hold the canonical 8 + 11 talks (stm + 16 kHz NIST SPHERE) in a flat <split>/ layout, which
# this job rearranges into the OpenSLR TEDLIUM_release-3/legacy/<split>/{stm,sph}/ layout the
# prep script and the ESPnet tedlium3 recipe expect. sha256 pinned to the mirror's LFS objects
# (checked 2026-09-02). spaCy NER on the lowercased references under-recalls (docs/06 entry 24).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
export XDG_CACHE_HOME="$work_root/cache/xdg"

raw=data/raw/tedlium3
mirror="https://huggingface.co/datasets/kfajdsl/tedlium/resolve/main/TEDLIUM_release3/legacy"
declare -A sha=(
  [dev]=4333d359ec3a7271b1fae03acd22a49de7b6cf293835efeef4a0c4c04fa192d6
  [test]=6be5e49a4dfc11ca4f4404acfefeadea3dda02f6d615342df4bb2e3cc37e530d
)
mkdir -p "$raw"
for split in dev test; do
  dst="$raw/TEDLIUM_release-3/legacy/$split"
  if [ -d "$dst/stm" ]; then
    echo "$dst already prepared, skipping download"
    continue
  fi
  wget -nv -O "$raw/$split.tar.gz" "$mirror/$split.tar.gz"
  echo "${sha[$split]}  $raw/$split.tar.gz" | sha256sum -c -
  tar xzf "$raw/$split.tar.gz" -C "$raw"
  mkdir -p "$dst/stm" "$dst/sph"
  mv "$raw/$split"/*.stm "$dst/stm/"
  mv "$raw/$split"/*.sph "$dst/sph/"
  rmdir "$raw/$split"
done

uv run python scripts/data/prepare_tedlium.py
uv run python scripts/data/build_ref_entity_index.py --manifest data/derived/tedlium3/manifest.jsonl \
  --out-dir data/derived/tedlium3/ref_entities --times-dir data/derived/tedlium3/ref_words
uv run python scripts/data/segment_vad.py --manifest data/derived/tedlium3/manifest.jsonl --out-dir data/derived/tedlium3/vad
