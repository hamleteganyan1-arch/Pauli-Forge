#!/usr/bin/env bash
#SBATCH --job-name=PauliForge-CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=pauliforge_cpu_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the PauliForge checkout}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
: "${PAULIFORGE_ENV:=$PWD/.venv/bin/activate}"
source "$PAULIFORGE_ENV"
pauliforge train "${PAULIFORGE_DATA:-data/processed}" \
  --config configs/experiment.kinetic.json --device cpu \
  --domain "${PAULIFORGE_DOMAIN:-configs/frozen_de.example.json}" \
  --output "${PAULIFORGE_OUTPUT:-results/model-cpu.pt}"
