#!/bin/bash
#SBATCH --job-name=spe-logprob
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# The 32 runs of the log-probability analysis, one per array task and GPU: task i runs
# scripts/shortcut_analysis.py with line i of `scripts/logprob_runs.py` (its Hydra overrides).
# Submit from the repository root, after `uv sync`:
#
#   sbatch --array=0-31 scripts/command_array.sh
#
# Models are read from HF_HOME (default Hugging Face cache if unset); add HF_HUB_OFFLINE=1 to the
# environment if the weights are already downloaded. Runs log to your W&B entity (WANDB_ENTITY or default).

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
ARGS=$(uv run --no-sync --project "$REPO" python "$REPO/scripts/logprob_runs.py" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$ARGS" ] || { echo "no run for task ${SLURM_ARRAY_TASK_ID}"; exit 1; }

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Hydra writes its outputs/ into the cwd: run from a private dir that links the repo's directories.
WORKDIR=$(mktemp -d "/tmp/spe_logprob_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_XXXX")
for d in "$REPO"/*/; do ln -s "$d" "$WORKDIR/$(basename "$d")"; done
cd "$WORKDIR"
echo "host: $(hostname) | job: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} | args: ${ARGS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# shellcheck disable=SC2086  # ARGS is a list of Hydra overrides
uv run --no-sync --project "$REPO" python "$REPO/scripts/shortcut_analysis.py" $ARGS
