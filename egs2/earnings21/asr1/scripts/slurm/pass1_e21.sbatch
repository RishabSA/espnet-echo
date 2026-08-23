#!/bin/bash
#SBATCH --job-name=e21-wlv3-pass1
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000M
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

# M2.5: full pass-1 decode (test + dev), word alignment, and the WER gate.
# pass1.py resumes per doc under --force, so if this hits the wall clock,
# resubmit the same script and it continues where it stopped.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
# openai-whisper checkpoints land in $XDG_CACHE_HOME/whisper; home is too small
export XDG_CACHE_HOME="$work_root/cache/xdg"

nvidia-smi --query-gpu=name,memory.total --format=csv

# _espnet suffix: the ESPnet-stack redo of the M2 decode (mentor directive 4); the
# HF-stack run e21_wlv3_pass1 stays untouched as the reference
uv run python scripts/decode/pass1.py --run-dir runs/e21_wlv3_pass1_espnet --split test --force
uv run python scripts/decode/pass1.py --run-dir runs/e21_wlv3_pass1_espnet --split dev --force
# --force: alignment is cheap and rerunning it keeps words.jsonl consistent with
# the current classifier after any align_words fix
uv run python scripts/decode/align_words.py --run-dir runs/e21_wlv3_pass1_espnet --force
uv run python scripts/eval/quick_wer.py --run-dir runs/e21_wlv3_pass1_espnet --split test
