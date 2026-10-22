#!/bin/bash -l
#SBATCH --job-name=PauliForge
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=pauliforge_%j.out
#SBATCH --error=pauliforge_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
# Activate YOUR environment explicitly before running.
: "${PAULIFORGE_ENV:?Set PAULIFORGE_ENV to the absolute path of your venv activate script}"
source "$PAULIFORGE_ENV"
pauliforge train data/processed --epochs 50 --domain configs/frozen_de.example.json --output model.pt
