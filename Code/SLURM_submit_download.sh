#!/bin/bash
#SBATCH --job-name=era5_download
#SBATCH --partition=thin                  # Snellius thin partition for I/O jobs
#SBATCH --time=02:00:00                   # 2h per job — generous for CDS latency
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                 # download is single-threaded
#SBATCH --mem=4G
#SBATCH --array=0-386%20                  # 387 jobs (9 vars × 43 years), max 20 concurrent
                                          # CDS allows ~20 simultaneous requests
#SBATCH --output=logs/download_%A_%a.out
#SBATCH --error=logs/download_%A_%a.err

# ── Load environment ──────────────────────────────────────────────────────────
module load 2022
module load Anaconda3/2022.05
source activate your_env_name            # adjust to your conda environment

# ── Derive VAR and YEAR from array task ID ────────────────────────────────────
# Task IDs 0-386 map to all combinations of 9 variables × 43 years
VARS=(t2m d2m u10 v10 u100 v100 ssrd swvl1 swvl2)
N_VARS=${#VARS[@]}                        # 9
YEAR_START=1980

VAR_IDX=$(( SLURM_ARRAY_TASK_ID / 43 ))  # integer division → variable index 0-8
YEAR_IDX=$(( SLURM_ARRAY_TASK_ID % 43 )) # remainder        → year offset 0-42

export VAR=${VARS[$VAR_IDX]}
export YEAR=$(( YEAR_START + YEAR_IDX ))

echo "Task ${SLURM_ARRAY_TASK_ID}: VAR=${VAR}, YEAR=${YEAR}"

# ── Create log directory if needed ───────────────────────────────────────────
mkdir -p logs

# ── Run download ──────────────────────────────────────────────────────────────
python 00_download_ERA5.py