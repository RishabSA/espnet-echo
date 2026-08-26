#!/bin/bash
#SBATCH --job-name=e1-e21
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000M
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

# M3b / E1: the GPU interventions of the decomposition study (docs/02 section 5,
# docs/08 M3b.2 to M3b.4, pre-registered in runs/_notes/gate0-prereg.md).
# Needs runs/e1/manifests (built on the laptop by e1_windows.py and rsynced here)
# and the base run runs/e21_wlv3_pass1_espnet. Every pass1.py call resumes per
# doc under --force, so a wall-clock hit is fixed by resubmitting this script.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
export XDG_CACHE_HOME="$work_root/cache/xdg"

nvidia-smi --query-gpu=name,memory.total --format=csv
test -d runs/e1/manifests/base || { echo "runs/e1/manifests missing: run e1_windows.py and rsync runs/e1 first"; exit 1; }

# boundary shifts (M3b.2): neighbourhood chunks moved by delta seconds
for cond in shift2 shift5 shift10; do
  uv run python scripts/decode/pass1.py --run-dir "runs/e1_$cond" --vad-dir "runs/e1/manifests/$cond" --split test --force
done

# decoder conditions (M3b.4): greedy, and a beam-8 repeat for byte identity
uv run python scripts/decode/pass1.py --run-dir runs/e1_greedy --vad-dir runs/e1/manifests/base --split test --num-beams 1 --num-return-sequences 1 --force
uv run python scripts/decode/pass1.py --run-dir runs/e1_repeat --vad-dir runs/e1/manifests/base --split test --force

# context windows (M3b.3): span only, +5 s, +15 s, each with and without the preceding pass-1 text as prompt
for cond in ctx_span ctx_5 ctx_15 ctx_span_prompt ctx_5_prompt ctx_15_prompt; do
  uv run python scripts/decode/pass1.py --run-dir "runs/e1_$cond" --vad-dir "runs/e1/manifests/$cond" --split test --force
done

# model sizes (intervention 5): full pass 1 on test, aligned, so evaluate.py can score them
for model in medium small; do
  run="runs/e21_w${model}_pass1_espnet"
  uv run python scripts/decode/pass1.py --run-dir "$run" --model "$model" --split test --force
  uv run python scripts/decode/align_words.py --run-dir "$run" --force
  uv run python scripts/eval/quick_wer.py --run-dir "$run" --split test
done
