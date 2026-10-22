#!/usr/bin/env bash
#SBATCH --job-name=PauliForge-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=pauliforge_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the PauliForge checkout}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
# Request the site's H100 partition with sbatch --partition=... if required.
: "${PAULIFORGE_ENV:=$PWD/.venv/bin/activate}"
source "$PAULIFORGE_ENV"
pauliforge train "${PAULIFORGE_DATA:-data/processed}" \
  --config configs/experiment.kinetic.json --device cuda \
  --domain "${PAULIFORGE_DOMAIN:-configs/frozen_de.example.json}" \
  --output "${PAULIFORGE_OUTPUT:-results/model-gpu.pt}"
