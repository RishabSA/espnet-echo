#!/bin/bash
#SBATCH --job-name=e21-prep
#SBATCH --partition=cpu
#SBATCH --account=bbjs-delta-cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=slurm-%x-%j.out

# corpus prep + model predownload, all CPU: run once before the GPU job.
# submit from the repo root on /work/hdd so SLURM_SUBMIT_DIR is the repo.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"

command -v ffmpeg >/dev/null || module load ffmpeg

uv run python scripts/data/prepare_earnings.py
uv run python scripts/data/prepare_conec.py
uv run python scripts/data/segment_vad.py
uv run python scripts/analysis/nc_distribution.py

uv run python -m nltk.downloader -d "$NLTK_DATA" averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict

# pull model weights here on the cpu partition so the gpu job never waits on I/O
uv run python - << 'EOF'
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

WhisperProcessor.from_pretrained("openai/whisper-large-v3")
WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", dtype=torch.float16)
torchaudio.pipelines.MMS_FA.get_model(with_star=True)
print("models cached")
EOF
