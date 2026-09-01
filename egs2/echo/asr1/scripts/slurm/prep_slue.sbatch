#!/bin/bash
#SBATCH --job-name=prep-slue
#SBATCH --partition=cpu
#SBATCH --account=bbjs-delta-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000M
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out

# SLUE-VoxPopuli NER subset: utterance-level documents for entity-F1 calibration (docs/03);
# SLUE fine-tune -> our dev, SLUE dev -> our test (the SLUE test set is blind). prepare_slue
# writes one-chunk vad manifests itself, so there is no segment_vad step.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
export XDG_CACHE_HOME="$work_root/cache/xdg"

mkdir -p data/raw/slue
if [ ! -d data/raw/slue/slue-voxpopuli ]; then
  wget -q "https://public-dataset-model-store.awsdev.asapp.com/users/sshon/public/slue/slue-voxpopuli_v0.2_blind.zip" -O data/raw/slue/slue-voxpopuli.zip
  unzip -q -o data/raw/slue/slue-voxpopuli.zip -d data/raw/slue
fi
uv run python scripts/data/prepare_slue.py
