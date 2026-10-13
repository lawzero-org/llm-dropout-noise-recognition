#!/bin/bash
#SBATCH --job-name=spe-agent
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# One in-process sweep agent per array task, one GPU each. Every agent loads the
# sweep's (fixed) model once and drains trials from the shared W&B queue, so the
# array size is the concurrency, not the run count.
#
#   sbatch --array=0-15 \
#     --export=ALL,SWEEP_ID=<entity>/<project>/<id>[,HF_HOME=<cache>] \
#     scripts/sweep_agents.sh
#
# Submit from the repository root. Models are read from HF_HOME (default Hugging Face cache
# if unset) in offline mode, so download them first (scripts/download_models.py); the pinned
# revision is a full commit hash, so offline lookup resolves the snapshot directly.
# Run `uv sync` once before submitting: the agents start with --no-sync so a wide array does
# not rewrite a shared venv concurrently.

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${SWEEP_ID:?pass SWEEP_ID=entity/project/id via --export}"
EXPERIMENT="${EXPERIMENT:-localization}"
CONFIG_NAME="${CONFIG_NAME:-localization}"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1  # stdout is block-buffered to the .out file otherwise; wandb's stderr is not
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

cd "$REPO"
echo "host: $(hostname)  |  job: ${SLURM_JOB_ID}  |  task: ${SLURM_ARRAY_TASK_ID:-0}  |  sweep: ${SWEEP_ID}"
echo "HF_HOME: ${HF_HOME:-default}  |  experiment: ${EXPERIMENT}  |  config: ${CONFIG_NAME}  |  code: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

uv run --no-sync spe-sweep --sweep-id "$SWEEP_ID" --config-name "$CONFIG_NAME" --experiment "$EXPERIMENT"
