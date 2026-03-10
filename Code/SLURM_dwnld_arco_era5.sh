#!/bin/bash
#SBATCH --job-name=era5_download
#SBATCH --array=1980-2024          # One task per year — adjust range to match YEAR_START/YEAR_END
#SBATCH --output=logs/era5_%a.out  # stdout per year (%a = array index = year)
#SBATCH --error=logs/era5_%a.err   # stderr per year
#SBATCH --partition=roma           # Snellius partition (roma = general purpose)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # gcsfs benefits from a few threads for parallel chunk fetching
#SBATCH --mem=8G                   # ~2 vars × 8760 h × 57 lat × 85 lon × float32 ≈ 320 MB raw; 8 GB is comfortable
#SBATCH --time=02:00:00            # 2 h per year is generous; typical run is 15–40 min
#SBATCH --export=ALL               # inherit full environment from ~/.bashrc, including ERA5_dat

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Load a Python module (check available versions with: module spider Python)
module load 2025
module load Miniforge3/25.3.0-3
eval "$(mamba shell hook --shell bash)"
mamba activate ~/.conda/envs/CE_env


LOG_DIR=$SLURM_SUBMIT_DIR/logs

mkdir -p "$LOG_DIR"
mkdir -p "$ERA5_dat" 

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# SLURM_ARRAY_TASK_ID is automatically set to the year (1980, 1981, … 2024)
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================"
echo "SLURM job:    $SLURM_JOB_ID"
echo "Array task:   $SLURM_ARRAY_TASK_ID  (year)"
echo "Node:         $SLURMD_NODENAME"
echo "Output dir:   $ERA5_dat"
echo "Start:        $(date)"
echo "========================================"

python 00_dwnld_arco_era5.py --year "$SLURM_ARRAY_TASK_ID"

EXIT_CODE=$?

echo "========================================"
echo "End:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE