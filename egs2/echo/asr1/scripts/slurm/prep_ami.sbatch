#!/bin/bash
#SBATCH --job-name=prep-ami
#SBATCH --partition=cpu
#SBATCH --account=bbjs-delta-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=30000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

# AMI IHM (Mix-Headset), standard full-corpus ASR dev+eval meetings only (~20 h, docs/03);
# entity index via spaCy projection with word times and speakers from the manual annotations.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
work_root="/work/hdd/bbjs/$USER"
export HF_HOME="$work_root/cache/huggingface"
export TORCH_HOME="$work_root/cache/torch"
export NLTK_DATA="$work_root/cache/nltk_data"
export UV_CACHE_DIR="$work_root/cache/uv"
export XDG_CACHE_HOME="$work_root/cache/xdg"

mkdir -p data/raw/ami
if [ ! -d data/raw/ami/annotations/words ]; then
  wget -q "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip" -O data/raw/ami/ami_public_manual.zip
  unzip -q -o data/raw/ami/ami_public_manual.zip -d data/raw/ami/annotations
fi
for m in $(uv run python -c "from scripts.data.prepare_ami import dev_meetings, eval_meetings; print(' '.join(dev_meetings + eval_meetings))"); do
  f="data/raw/ami/amicorpus/$m/audio/$m.Mix-Headset.wav"
  if [ ! -f "$f" ]; then
    mkdir -p "data/raw/ami/amicorpus/$m/audio"
    wget -q "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/$m/audio/$m.Mix-Headset.wav" -O "$f"
  fi
done
uv run python scripts/data/prepare_ami.py
uv run python scripts/data/build_ref_entity_index.py --manifest data/derived/ami/manifest.jsonl \
  --out-dir data/derived/ami/ref_entities --times-dir data/derived/ami/ref_words
uv run python scripts/data/segment_vad.py --manifest data/derived/ami/manifest.jsonl --out-dir data/derived/ami/vad
